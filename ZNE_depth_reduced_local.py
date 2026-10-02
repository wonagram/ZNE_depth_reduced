from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    coherent_unitary_error
)

from qiskit.quantum_info import Operator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import matplotlib.pyplot as plt
from U_config import *
from state_preparation import *

from local_folding_helpers import (
    make_local_plan,
    build_intermediate_local_circuit,
    check_intermediate_local_folding,
    describe_local_plan,
)

# ============================================================
# Hardware / simulator

# If you can specify a device. If it is false you use the least busy one.
# ============================================================

USE_REAL_HARDWARE = False
DEVICE_NAME = False
# DEVICE_NAME = "ibm_strasbourg"

shots = 10**3

# ============================================================
# Depth of the unitary U
# ============================================================

DEPTH = 4


# ============================================================
# Number of qubits in U
# ============================================================

NUM_U_QUBITS = 2

# Physical pool = NUM_U_QUBITS * fold factor, independent of DEPTH.
# "fixed": permanent lanes (original r*n+k in front input placement).
# "dynamic": all currently free/reset workspace is shared between logical qubits.
QUBIT_ASSIGNMENT = "dynamic"

# "front": input state on q0,...,q(n-1), preserving the previous four modes.
# "first_use": spatial input placement follows first use by a component.
# The full (possibly entangled) input is still prepared once at the start.
INPUT_PLACEMENT = "first_use"

# "snake": fixed lanes reverse per use; dynamic blocks alternate down/up.
# "cycle": fixed lanes wrap; dynamic workspace restarts at the top each time.
REUSE_PATTERN = "cycle"

# ============================================================
# Circuit drawing
# If this flag is turn on, it draws only circuit and exit the code.
# ============================================================

DRAW_CIRCUIT = True
DRAW_DEPTH_FOLDED = 3

# ============================================================
# Exact, noiseless branch checking
# ============================================================

CHECK_CODE = False
CHECK_DEPTH_FOLDED = 3

CHECK_MAX_PHYSICAL_QUBITS = 20
CHECK_MAX_BRANCHES = 4096
CHECK_PRINT_EACH = True



# Configurable guard against accidentally constructing an
# impractically wide circuit for ordinary local simulation.
MAX_SIMULATOR_PHYSICAL_QUBITS = 24

# ============================================================
# Number of foldings
# ============================================================

depth_folded_circuits = [1, 3, 5, 7]



# ============================================================
# Noise flags for a simulator

# Turn off all noise when drawing circuit
# ============================================================

USE_DEPOLARIZING = True
USE_AMPLITUDE_DAMPING = False
USE_PHASE_DAMPING = False
USE_COHERENT_OVERROTATION = False

if DRAW_CIRCUIT or CHECK_CODE:
    USE_DEPOLARIZING = False
    USE_AMPLITUDE_DAMPING = False
    USE_PHASE_DAMPING = False
    USE_COHERENT_OVERROTATION = False

# ============================================================
# Noise parameters for a simulator
# ============================================================

depolarizing_strength = 0.02
amplitude_damping_strength = 0.02
phase_damping_strength = 0.02

# coherent over-rotation angle
overrotation_epsilon = 0.02


if NUM_U_QUBITS not in U_GATES:
    raise ValueError(
        f"Unsupported NUM_U_QUBITS = {NUM_U_QUBITS}"
    )

U_layers = U_GATES[NUM_U_QUBITS]
if not 1 <= DEPTH <= len(U_layers):
    raise ValueError(
        f"DEPTH must be between 1 and {len(U_layers)}, got {DEPTH}."
    )

# ============================================================
# Noise instructions
# ============================================================

depolarizing_noise_1q = depolarizing_error(
    depolarizing_strength,
    1
).to_instruction()

depolarizing_noise_2q = depolarizing_error(
    depolarizing_strength,
    2
).to_instruction()

amplitude_damping_noise = amplitude_damping_error(
    amplitude_damping_strength,
    canonical_kraus=False
).to_instruction()

phase_damping_noise = phase_damping_error(
    phase_damping_strength,
    canonical_kraus=False
).to_instruction()

# ============================================================
# Apply selected noise channels
# ============================================================

def apply_noise(qc, wires, gate_type):

    # ========================================================
    # 1-qubit gate noise
    # ========================================================

    if gate_type in ["x", "y", "z"]:

        wire = wires[0]

        if USE_DEPOLARIZING:
            qc.append(
                depolarizing_noise_1q,
                [wire]
            )

        if USE_AMPLITUDE_DAMPING:
            qc.append(
                amplitude_damping_noise,
                [wire]
            )

        if USE_PHASE_DAMPING:
            qc.append(
                phase_damping_noise,
                [wire]
            )

        if USE_COHERENT_OVERROTATION:

            error_circuit = QuantumCircuit(1)

            if gate_type == "x":
                error_circuit.rx(overrotation_epsilon, 0)

            elif gate_type == "y":
                error_circuit.ry(overrotation_epsilon, 0)

            elif gate_type == "z":
                error_circuit.rz(overrotation_epsilon, 0)

            coherent_noise = coherent_unitary_error(
                Operator(error_circuit)
            ).to_instruction()

            qc.append(
                coherent_noise,
                [wire]
            )


    # ========================================================
    # 2-qubit CNOT noise
    # ========================================================

    elif gate_type == "cx":

        control = wires[0]
        target = wires[1]

        # ----------------------------------------------------
        # 2-qubit depolarizing noise
        # ----------------------------------------------------
        if USE_DEPOLARIZING:
            qc.append(
                depolarizing_noise_2q,
                [control, target]
            )

        # ----------------------------------------------------
        # Independent amplitude damping on both qubits
        # ----------------------------------------------------
        if USE_AMPLITUDE_DAMPING:
            qc.append(
                amplitude_damping_noise,
                [control]
            )

            qc.append(
                amplitude_damping_noise,
                [target]
            )

        # ----------------------------------------------------
        # Independent phase damping on both qubits
        # ----------------------------------------------------
        if USE_PHASE_DAMPING:
            qc.append(
                phase_damping_noise,
                [control]
            )

            qc.append(
                phase_damping_noise,
                [target]
            )

def apply_rotation(qc, wire, axis, angle):

    if axis == "x":
        qc.rx(angle, wire)

    elif axis == "y":
        qc.ry(angle, wire)

    elif axis == "z":
        qc.rz(angle, wire)


def applyU(qc, wires, block_index):

    U_layers = U_GATES[NUM_U_QUBITS]

    if not 1 <= DEPTH <= len(U_layers):
        raise ValueError(
            f"DEPTH must be between 1 and {len(U_layers)}, got {DEPTH}."
        )

    # Python block_index:
    # 0 -> block 1 : normal U
    # 1 -> block 2 : YY around every CNOT
    # 2 -> block 3 : normal U
    # 3 -> block 4 : YY around every CNOT
    is_even_block = (block_index % 2 == 1)

    for layer in U_layers[:DEPTH]:

        for gate in layer:

            gate_type = gate[0]

            # ====================================================
            # Single-qubit rotation
            # ====================================================
            if gate_type in ["x", "y", "z"]:

                _, local_qubit, theta = gate

                wire = wires[local_qubit]

                apply_rotation(
                    qc,
                    wire,
                    gate_type,
                    theta
                )

                if not USE_REAL_HARDWARE:
                    apply_noise(
                        qc,
                        [wire],
                        gate_type
                    )

            # ====================================================
            # CNOT
            # ====================================================
            elif gate_type == "cx":

                _, control, target = gate

                control_wire = wires[control]
                target_wire = wires[target]

                # ------------------------------------------------
                # Even logical block:
                #
                # (Y x Y) CNOT (Y x Y)
                # ------------------------------------------------
                if is_even_block:

                    qc.y(control_wire)
                    qc.y(target_wire)

                # CNOT
                qc.cx(
                    control_wire,
                    target_wire
                )

                if not USE_REAL_HARDWARE:
                    apply_noise(
                        qc,
                        [control_wire, target_wire],
                        "cx"
                    )

                # ------------------------------------------------
                # Even logical block:
                # second Y x Y
                # ------------------------------------------------
                if is_even_block:

                    qc.y(control_wire)
                    qc.y(target_wire)

            else:
                raise ValueError(
                    f"Unknown gate type: {gate_type}"
                )


def get_block(q, block_index):

    start = block_index * NUM_U_QUBITS
    end = start + NUM_U_QUBITS

    return [
        q[i]
        for i in range(start, end)
    ]

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
    # 00 -> XZ
    with qc.if_test((c_pair, 0)):
        qc.z(target)
        qc.x(target)

    # 01 -> Z
    with qc.if_test((c_pair, 1)):
        qc.z(target)

    # 10 -> X
    with qc.if_test((c_pair, 2)):
        qc.x(target)

    # 11 -> I
    # do nothing


folded_circuits = []

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
        U_layers[:DEPTH],
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
        apply_rotation=apply_rotation,
        apply_noise=apply_noise,
        psi_minus=psi_minus,
        bell_measure=bell_measure,
        apply_correction=apply_correction,
        use_noise=(
            not USE_REAL_HARDWARE
            and not DRAW_CIRCUIT
            and not CHECK_CODE
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


# ============================================================
# Backend
# ============================================================


if USE_REAL_HARDWARE:

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

print(f"\nDEPTH = {DEPTH}")

if USE_REAL_HARDWARE:
    print(f"Backend: {backend.name}")

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

print(f"\nExpectation values of local method:\n"
      f"{[round(x, 5) for x in expectation_values]}")


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
        f"{zero_noise:.6f}"
    )
