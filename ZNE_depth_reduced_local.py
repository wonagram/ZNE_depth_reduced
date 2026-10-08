from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np
import matplotlib.pyplot as plt

from U_config import (
    get_unitary_max_depth,
    normalize_unitary_type,
    select_unitary_components,
    selected_unitary_summary,
)
from unitary_tools import (
    NoiseController,
    append_local_component_copy,
    selected_component_matrix,
)
from state_preparation import *

from local_folding_helpers import (
    make_local_plan,
    build_intermediate_local_circuit,
    check_intermediate_local_folding,
    describe_local_plan,
)

# ============================================================
# Hardware / simulator / statevector
# ============================================================
USE_REAL_HARDWARE = False
DEVICE_NAME = False
# DEVICE_NAME = "ibm_strasbourg"

USE_STATEVECTOR = False
shots = 10**3  # Used only when USE_STATEVECTOR is False.

# ============================================================
# Unitary selection, counted depth, and logical-qubit count
# Available UNITARY_TYPE values:
#   random_type, hea_cz, hea_crx, hea_crz, hea_rzz,
#   qaoa_ising, qaoa_xy, clifford_only
# ============================================================
UNITARY_TYPE = "qaoa"
DEPTH = 31
NUM_U_QUBITS = 2

# Physical pool = NUM_U_QUBITS * fold factor, independent of DEPTH.
# "fixed": permanent lanes. "dynamic": shared free/reset workspace.
QUBIT_ASSIGNMENT = "dynamic"

# "front": input state on q0,...,q(n-1).
# "first_use": spatial placement follows first component use.
INPUT_PLACEMENT = "first_use"

# "snake": alternating directions. "cycle": restart from the top.
REUSE_PATTERN = "cycle"

# ============================================================
# Circuit drawing
# ============================================================
DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 3

# ============================================================
# Exact, noiseless branch checking
# ============================================================
CHECK_CODE = False
CHECK_DEPTH_FOLDED = 3
CHECK_MAX_PHYSICAL_QUBITS = 20
CHECK_MAX_BRANCHES = 4096
CHECK_PRINT_EACH = True

# Guard against accidentally constructing an impractically wide simulator job.
MAX_SIMULATOR_PHYSICAL_QUBITS = 24

# ============================================================
# Number of foldings
# ============================================================
depth_folded_circuits = [1, 3, 5, 7]

# ============================================================
# Noise flags and strengths
# ============================================================
USE_DEPOLARIZING = True
USE_AMPLITUDE_DAMPING = False
USE_PHASE_DAMPING = False
USE_COHERENT_OVERROTATION = False

depolarizing_strength = 0.02
amplitude_damping_strength = 0.02
phase_damping_strength = 0.02
overrotation_epsilon = 0.02

if USE_STATEVECTOR:
    USE_REAL_HARDWARE = False

if USE_STATEVECTOR or DRAW_CIRCUIT or CHECK_CODE:
    USE_DEPOLARIZING = False
    USE_AMPLITUDE_DAMPING = False
    USE_PHASE_DAMPING = False
    USE_COHERENT_OVERROTATION = False

# ============================================================
# Select the exact counted-depth prefix.
# A cut RXX/RYY/RZZ prefix remains one atomic local component.
# ============================================================
UNITARY_TYPE = normalize_unitary_type(UNITARY_TYPE)
SELECTED_COMPONENTS = select_unitary_components(
    UNITARY_TYPE,
    NUM_U_QUBITS,
    DEPTH,
)
MAX_U_DEPTH = get_unitary_max_depth(UNITARY_TYPE, NUM_U_QUBITS)

NOISE_CONTROLLER = NoiseController(
    use_depolarizing=USE_DEPOLARIZING,
    use_amplitude_damping=USE_AMPLITUDE_DAMPING,
    use_phase_damping=USE_PHASE_DAMPING,
    use_coherent_overrotation=USE_COHERENT_OVERROTATION,
    depolarizing_strength=depolarizing_strength,
    amplitude_damping_strength=amplitude_damping_strength,
    phase_damping_strength=phase_damping_strength,
    overrotation_epsilon=overrotation_epsilon,
)


def apply_selected_component(qc, mapped_wires, component, block_index, use_noise):
    """Fold the complete selected component, not its decomposition pieces."""
    append_local_component_copy(
        qc,
        mapped_wires,
        component,
        block_index=block_index,
        noise_controller=NOISE_CONTROLLER,
        use_noise=use_noise,
    )


def psi_minus(qc, wires):
    qc.h(wires[0])
    qc.cx(wires[0], wires[1])
    qc.z(wires[0])
    qc.x(wires[1])


def bell_measure(qc, wires):
    qc.cx(wires[0], wires[1])
    qc.h(wires[0])


def apply_correction(qc, c_pair, target):
    """
    Bell outcome:
        00 -> XZ
        01 -> Z
        10 -> X
        11 -> I
    """
    with qc.if_test((c_pair, 0)):
        qc.z(target)
        qc.x(target)
    with qc.if_test((c_pair, 1)):
        qc.z(target)
    with qc.if_test((c_pair, 2)):
        qc.x(target)

# ============================================================
# Noiseless statevector execution with intermediate qubit reuse
# ============================================================
# These functions read the ACTUAL circuit emitted by the existing helper.
# They do not replace a folded component with its ideal logical gate.
# Measurement records are kept coherently within each component. Before
# resetting/reusing their wires, we verify that they factor from ALL other
# wires. Only then is the record factor replaced by |0...0>.
# No density matrix, noise trajectories, or measurement samples are used.


def _sv_preparation(circuit, prep_end):
    """Convert a pure input preparation to a measurement-free circuit."""
    from qiskit.circuit import Gate

    result = QuantumCircuit(circuit.num_qubits)
    result.global_phase = circuit.global_phase
    touched = set()
    for item in circuit.data[:prep_end]:
        op = item.operation
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == "barrier":
            continue
        if item.clbits or getattr(op, "condition", None) is not None:
            raise ValueError("Statevector input preparation must not use classical bits.")
        if op.name in ("initialize", "reset"):
            # Initialize normally includes reset. It is safely unitary here
            # ONLY on previously untouched wires, which are known to be |0>.
            if touched.intersection(wires):
                raise ValueError(
                    "Statevector mode needs pure input preparation. initialize/reset "
                    "on previously used wires is not silently sampled. Use a unitary "
                    "preparation or initialize each input once on fresh |0> wires."
                )
            if op.name == "reset":
                continue
            prep_gate = op.gates_to_uncompute().inverse().to_gate()
            result.append(prep_gate, wires)
        elif isinstance(op, Gate):
            result.append(op, wires)
        else:
            raise ValueError(
                f"Unsupported input instruction {op.name!r} in statevector mode. "
                "Use unitary gates or initialize on untouched input wires."
            )
        touched.update(wires)
    return result


def _sv_coherent_component(circuit, record, stage):
    """Translate the component's actual measurement/if_test records coherently.

    The current helper uses independent (register, integer) if_test blocks
    containing Pauli gates and no else branches. Unsupported control flow is
    rejected rather than ignored. Measurement controls cannot be overwritten.
    """
    from qiskit.circuit import Gate

    width = circuit.num_qubits
    result = QuantumCircuit(width)
    measured = {}  # Actual classical-bit object -> physical measurement wire.
    correction_started = False
    start, stop = record["start"], record["corrections_end"]
    for item in circuit.data[start:stop]:
        op = item.operation
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == "barrier":
            continue
        if op.name == "measure":
            if correction_started or len(wires) != 1 or len(item.clbits) != 1:
                raise ValueError("A component must measure all Bell wires before correcting.")
            bit = item.clbits[0]
            if bit in measured or wires[0] in measured.values():
                raise ValueError("A Bell bit/wire is measured twice in the same component.")
            measured[bit] = wires[0]
            continue
        if op.name == "if_else":
            correction_started = True
            condition = op.condition
            if not isinstance(condition, tuple) or len(condition) != 2:
                raise ValueError("Statevector mode requires (classical bit/register, value) conditions.")
            classical, value = condition
            bits = list(classical) if isinstance(classical, ClassicalRegister) else [classical]
            if not bits or any(bit not in measured for bit in bits):
                raise ValueError("A correction depends on a bit outside this component's Bell records.")
            controls = [measured[bit] for bit in bits]
            value = int(value)
            if not 0 <= value < (1 << len(controls)):
                raise ValueError("Classical correction value is outside its register range.")
            if len(op.blocks) > 1 and op.blocks[1] is not None:
                raise ValueError("This statevector path supports if_test corrections without else.")
            body = op.blocks[0]
            if abs(float(body.global_phase)) > 1e-14:
                raise ValueError("A phase-bearing conditional block needs an explicit coherent translation.")
            if len(body.qubits) != len(wires):
                raise ValueError("Unexpected if_test qubit mapping.")
            for inner in body.data:
                gate = inner.operation
                if gate.name == "barrier":
                    continue
                targets = [wires[body.find_bit(q).index] for q in inner.qubits]
                if (inner.clbits or not isinstance(gate, Gate)
                        or getattr(gate, "condition", None) is not None):
                    raise ValueError("Only unitary correction gates are supported inside if_test.")
                if set(targets).intersection(measured.values()):
                    raise ValueError("A correction must not overwrite a Bell-record wire.")
                # Register iteration is little-endian. ctrl_state=value
                # therefore uses the original c[1]c[0] convention exactly.
                result.append(
                    gate.control(num_ctrl_qubits=len(controls), ctrl_state=value),
                    controls + targets,
                )
            continue
        if measured or correction_started:
            raise ValueError("Unexpected unconditional gate after the component's Bell measurements.")
        if item.clbits or not isinstance(op, Gate) or getattr(op, "condition", None) is not None:
            raise ValueError(f"Non-unitary/unsupported component instruction: {op.name!r}.")
        result.append(op, wires)

    actual_resets = []
    for item in circuit.data[stop:record["end"]]:
        if item.operation.name == "barrier":
            continue
        if item.operation.name != "reset" or item.clbits or len(item.qubits) != 1:
            raise ValueError("Only workspace resets may follow the component's corrections.")
        actual_resets.append(circuit.find_bit(item.qubits[0]).index)
    if (actual_resets != list(record["reset_ids"])
            or actual_resets != list(stage["reset_ids"])
            or len(set(actual_resets)) != len(actual_resets)
            or set(actual_resets) != set(measured.values())):
        raise ValueError("Actual Bell measurements/resets do not match the reuse plan.")
    if set(actual_resets).intersection(record["output_ids"]):
        raise ValueError("A reset would erase a live logical output/spectator.")
    return result, actual_resets


def _sv_run_segment(segment, backend, initial=None, output_ids=None):
    """Evolve a measurement-free segment using genuine Aer statevector APIs."""
    run = QuantumCircuit(segment.num_qubits)
    if initial is not None:
        run.set_statevector(initial)
    run.compose(segment, inplace=True)
    run.save_statevector(label="local_statevector")
    if output_ids is not None:
        run.save_expectation_value(
            Pauli("Z" * len(output_ids)), output_ids, label="local_expectation"
        )
    compiled = transpile(run, backend, optimization_level=0)
    if compiled.num_qubits != segment.num_qubits:
        raise ValueError("Transpilation changed the physical register width.")
    if compiled.layout is not None:
        if compiled.layout.final_index_layout() != list(range(segment.num_qubits)):
            raise ValueError("Unexpected simulator layout: statevector wire order changed.")
    forbidden = {"measure", "reset", "if_else", "kraus", "superop", "quantum_channel"}
    if any(item.operation.name in forbidden for item in compiled.data):
        raise ValueError("A stochastic instruction reached the no-shot statevector backend.")
    # Aer uses the shots argument as an execution count. There are NO
    # measurements, resets, or noise here, so this is not a one-shot estimate.
    execution = backend.run(compiled, shots=1).result()
    if not execution.success:
        raise RuntimeError(f"Aer statevector execution failed: {execution.status}")
    data = execution.data(0)
    vector = np.asarray(data["local_statevector"], dtype=np.complex128).copy()
    norm_squared = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm_squared) or abs(norm_squared - 1.0) > 1e-9:
        raise ValueError(f"Invalid statevector norm squared: {norm_squared}.")
    expectation = None
    if output_ids is not None:
        value = complex(data["local_expectation"])
        if abs(value.imag) > 1e-10 or not np.isfinite(value.real):
            raise ValueError(f"Invalid real observable expectation: {value}.")
        expectation = float(value.real)
    return vector, expectation


def _sv_reset_factored_records(vector, reset_ids, width):
    """Discard a record factor ONLY after testing factorization of all amplitudes.

    Reshaping the vector into record/live indices uses 2**width amplitudes,
    NOT a 2**width by 2**width density matrix. A reference row identifies
    the candidate live vector, but EVERY row is checked, including phases.
    This is not postselection, branch sampling, or an unchecked approximation.
    """
    if not reset_ids:
        return vector, 0.0
    reset_ids = list(reset_ids)
    if len(set(reset_ids)) != len(reset_ids) or any(q < 0 or q >= width for q in reset_ids):
        raise ValueError("Invalid reset wire list.")
    keep_ids = [q for q in range(width) if q not in set(reset_ids)]
    axes = [width - 1 - q for q in reversed(reset_ids)]
    axes += [width - 1 - q for q in reversed(keep_ids)]
    grouped = vector.reshape((2,) * width).transpose(axes).reshape(1 << len(reset_ids), -1)
    row_norms = np.sum(np.abs(grouped) ** 2, axis=1)
    pivot = int(np.argmax(row_norms))
    norm_squared = float(np.sum(row_norms))
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("Invalid coherent state before workspace reset.")
    live = grouped[pivot].copy() / np.sqrt(row_norms[pivot])
    # Check all record sectors in bounded temporary buffers. The residual is
    # ||Psi - record_factor tensor live||_2 / ||Psi||_2.
    chunk_rows = max(1, (1 << 16) // grouped.shape[1])
    residual_squared = 0.0
    conjugate_live = live.conj()
    for start in range(0, grouped.shape[0], chunk_rows):
        block = grouped[start:start + chunk_rows]
        weights = block @ conjugate_live
        difference = block - weights[:, None] * live[None, :]
        residual_squared += float(np.vdot(difference, difference).real)
    residual = float(np.sqrt(max(0.0, residual_squared) / norm_squared))
    if residual > 1e-11:
        raise ValueError(
            f"Bell records remain correlated with live qubits (residual={residual:.3e}). "
            "Exact single-statevector reuse is not valid for this component. "
            "Check the circuit/corrections, use CHECK_CODE on a smaller case, "
            "or use USE_STATEVECTOR=False. No branch was sampled or discarded."
        )
    # All conditional live states agree up to phase to numerical precision.
    # Physical reset now replaces only the independent record factor by |0>.
    reset_grouped = np.zeros_like(grouped)
    reset_grouped[0] = live
    restored = reset_grouped.reshape((2,) * width).transpose(np.argsort(axes))
    return np.ascontiguousarray(restored).reshape(-1), residual


def local_statevector_expectation(circuit, plan, backend):
    """Run all actual folded components; return the no-shot output expectation.

    The metadata is produced by the user's existing local_folding_helpers.
    Rank-one record/live factorization is checked at EVERY reset boundary.
    Failure stops the run instead of returning a sampled or postselected value.
    """
    meta = (circuit.metadata or {}).get("intermediate_local_folding")
    required = {"prep_end", "stages", "output_start", "output_ids", "use_noise"}
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError("The local helper must provide intermediate_local_folding stage metadata.")
    if meta["use_noise"]:
        raise ValueError("Statevector mode requires use_noise=False when building the circuit.")
    if (circuit.num_qubits != plan["num_physical_qubits"]
            or len(meta["stages"]) != len(plan["stages"])
            or list(meta["output_ids"]) != list(plan["output_ids"])):
        raise ValueError("The circuit and local folding plan do not match.")
    out_registers = [register for register in circuit.cregs if register.name == "out"]
    if len(out_registers) != 1 or len(out_registers[0]) != plan["n"]:
        raise ValueError("Expected one final output register named out.")
    readout = list(circuit.data[meta["output_start"]:])
    if len(readout) != plan["n"]:
        raise ValueError("Unexpected operations after the final local component.")
    for k, item in enumerate(readout):
        if (item.operation.name != "measure" or len(item.qubits) != 1
                or len(item.clbits) != 1
                or circuit.find_bit(item.qubits[0]).index != plan["output_ids"][k]
                or item.clbits[0] != out_registers[0][k]):
            raise ValueError("The final readout does not match the logical output map.")

    previous_end = int(meta["prep_end"])
    if not 0 <= previous_end <= meta["output_start"] <= len(circuit.data):
        raise ValueError("Invalid local component boundaries.")
    preparation = _sv_preparation(circuit, previous_end)
    vector, expectation = _sv_run_segment(
        preparation, backend, output_ids=list(plan["input_ids"])
    )
    max_residual = 0.0
    for j, (record, stage) in enumerate(zip(meta["stages"], plan["stages"])):
        needed = {"layer", "start", "end", "corrections_end", "output_ids", "reset_ids"}
        if not needed.issubset(record):
            raise ValueError("The local helper's stage metadata is incomplete.")
        if (record["layer"] != j or record["start"] != previous_end
                or not record["start"] <= record["corrections_end"] <= record["end"] <= meta["output_start"]
                or list(record["output_ids"]) != list(stage["outputs_after_layer"])):
            raise ValueError(f"G{j + 1}: inconsistent component/output metadata.")
        coherent, reset_ids = _sv_coherent_component(circuit, record, stage)
        vector, expectation = _sv_run_segment(
            coherent, backend, initial=vector, output_ids=list(record["output_ids"])
        )
        try:
            vector, residual = _sv_reset_factored_records(vector, reset_ids, circuit.num_qubits)
        except ValueError as error:
            raise ValueError(f"fold={plan['fold_factor']}, G{j + 1}: {error}") from error
        max_residual = max(max_residual, residual)
        previous_end = record["end"]
    if previous_end != meta["output_start"]:
        raise ValueError("Some circuit operations were not included in the local components.")
    return expectation, max_residual


folded_circuits = []
local_plans = []

if CHECK_CODE:
    depths_to_build = [CHECK_DEPTH_FOLDED]
elif DRAW_CIRCUIT:
    depths_to_build = [DRAW_DEPTH_FOLDED]
else:
    depths_to_build = depth_folded_circuits


for depth_folded in depths_to_build:

    # --------------------------------------------------------
    # 1. Preserve the user-defined component/layer boundaries.
    #    Reuse reset wires with the chosen allocation/routing policies.
    # --------------------------------------------------------
    plan = make_local_plan(
        NUM_U_QUBITS,
        SELECTED_COMPONENTS,
        depth_folded,
        reuse_pattern=REUSE_PATTERN,
        qubit_assignment=QUBIT_ASSIGNMENT,
        input_placement=INPUT_PLACEMENT,
    )
    describe_local_plan(plan)

    if (
        not USE_REAL_HARDWARE
        and not DRAW_CIRCUIT
        and not CHECK_CODE
        and plan["num_physical_qubits"] > MAX_SIMULATOR_PHYSICAL_QUBITS
    ):
        raise ValueError(
            f"Local circuit uses {plan['num_physical_qubits']} physical qubits. "
            "Reduce NUM_U_QUBITS/fold factors or deliberately increase "
            "MAX_SIMULATOR_PHYSICAL_QUBITS after checking resources."
        )

    # --------------------------------------------------------
    # 2. For EACH G_i:
    #    copies -> all its Bell measurements -> corrections -> reset measured wires.
    #    Only after correcting do we append G_(i+1).
    #
    # The returned circuit ALSO includes the final out readout.
    # Do not call the old finish_local_circuit() on it.
    # --------------------------------------------------------
    qc = build_intermediate_local_circuit(
        plan,
        prepare_initial_state=prepare_initial_state,
        apply_component=apply_selected_component,
        psi_minus=psi_minus,
        bell_measure=bell_measure,
        apply_correction=apply_correction,
        use_noise=(
            not USE_REAL_HARDWARE
            and not DRAW_CIRCUIT
            and not CHECK_CODE
            and not USE_STATEVECTOR
        ),
        stage_barriers=True,
    )

    # --------------------------------------------------------
    # 3. Exact branch-by-branch dynamic-circuit check.
    #    The checker projects each intermediate measurement,
    #    follows actual if_test corrections and resets before reusing wires.
    #    No backend is selected and no circuit job is submitted.
    # --------------------------------------------------------
    if CHECK_CODE:
        check_report = check_intermediate_local_folding(
            qc,
            plan,
            prepare_initial_state=prepare_initial_state,
            component_matrix=selected_component_matrix,
            max_physical_qubits=CHECK_MAX_PHYSICAL_QUBITS,
            max_branches=CHECK_MAX_BRANCHES,
            print_each=CHECK_PRINT_EACH,
        )
        raise SystemExit(0 if check_report["passed"] else 1)

    # --------------------------------------------------------
    # 4. Draw the FULL circuit, including intermediate
    #    conditional Pauli corrections and final readout.
    # --------------------------------------------------------
    if DRAW_CIRCUIT:
        print("\nFull intermediate-measurement/correction circuit.")
        qc.draw(
            "mpl",
            style="clifford",
            plot_barriers=True,
            fold=-1
        )
        plt.show()
        raise SystemExit

    folded_circuits.append(qc)
    local_plans.append(plan)


# ============================================================
# Backend
# ============================================================


if USE_STATEVECTOR:
    backend = AerSimulator(
        method="statevector",
        precision="double",
        zero_threshold=0.0,
        enable_truncation=False,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
elif USE_REAL_HARDWARE:

    if not QiskitRuntimeService.saved_accounts():
        raise RuntimeError(
            "USE_REAL_HARDWARE=True, but no IBM Quantum account is saved."
        )

    service = QiskitRuntimeService(channel="ibm_quantum")

    if DEVICE_NAME:
        backend = service.backend(DEVICE_NAME)

    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            dynamic_circuits=True,
        )

else:

    backend = AerSimulator()
    backend.set_options(seed_simulator=150)




# ============================================================
# Transpilation
# ============================================================
if USE_STATEVECTOR:
    expectation_values = []
    for circuit, plan in zip(folded_circuits, local_plans):
        expectation, reset_residual = local_statevector_expectation(circuit, plan, backend)
        expectation_values.append(expectation)
        print(
            f"Statevector fold={plan['fold_factor']}: {expectation:.12f}; "
            f"max reset-factorization residual={reset_residual:.3e}"
        )
else:
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=0
    )

    exec_circuits = [
        pm.run(circuit)
        for circuit in folded_circuits
    ]

    sampler = Sampler(mode=backend)

    job = sampler.run(
        exec_circuits,
        shots=shots
    )

    result = job.result()

    # ============================================================
    # Z expectation value for n-qubit output
    #
    # <Z^{\otimes n}> = sum_{x in {0,1}^n} (-1)^{|x|} P(x)
    #
    # where |x| is the number of 1s in the bitstring x.
    # Therefore:
    #   even number of 1s -> +1
    #   odd  number of 1s -> -1
    # ============================================================

    expectation_values = []

    for i in range(len(folded_circuits)):

        out_counts = result[i].data.out.get_counts()

        total_counts = sum(out_counts.values())

        expectation = 0.0

        for bitstring, count in out_counts.items():

            # Remove spaces if present
            bitstring = bitstring.replace(" ", "")

            # Eigenvalue of Z^{\otimes n}:
            # even number of 1s -> +1
            # odd  number of 1s -> -1
            eigenvalue = (-1) ** bitstring.count("1")

            expectation += eigenvalue * count / total_counts

        expectation_values.append(expectation)


# ============================================================
# Print experiment configuration
# ============================================================

print(f"\nUNITARY_TYPE = {UNITARY_TYPE}")
print(selected_unitary_summary(UNITARY_TYPE, NUM_U_QUBITS, DEPTH))
print(f"DEPTH = {DEPTH}; maximum configured depth = {MAX_U_DEPTH}")

if USE_REAL_HARDWARE:
    print(f"Backend: {backend.name}")

else:
    if USE_STATEVECTOR:
        print("Backend: AerSimulator (statevector)")
    else:
        print("Backend: AerSimulator")

    active_noises = []

    if USE_DEPOLARIZING:
        active_noises.append("Depolarizing")

    if USE_AMPLITUDE_DAMPING:
        active_noises.append("Amplitude damping")

    if USE_PHASE_DAMPING:
        active_noises.append("Phase damping")

    if USE_COHERENT_OVERROTATION:
        active_noises.append("Coherent overrotation")

    if active_noises:
        print("Noise:", ", ".join(active_noises))
    else:
        print("Noise: none")

precision = 12 if USE_STATEVECTOR else 4
print("\nExpectation values of local method:")
print("[" + ", ".join(f"{x:.{precision}f}" for x in expectation_values) + "]")


# ============================================================
# Zero-noise extrapolation
# ============================================================
depth_folded_circuits_np = np.array(
    depth_folded_circuits,
    dtype=float
)

expectation_values_np = np.array(
    expectation_values,
    dtype=float
)


for degree in range(
    1,
    min(4, len(expectation_values) - 1) + 1,
):
    coeffs = np.polyfit(
        depth_folded_circuits_np,
        expectation_values_np,
        deg=degree
    )

    zero_noise = np.polyval(coeffs, 0.0)

    print(
        f"degree {degree}: "
        f"{zero_noise:.{precision}f}"
    )
