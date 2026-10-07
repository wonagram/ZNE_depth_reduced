from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    coherent_unitary_error
)
from qiskit.quantum_info import Operator, Pauli
from qiskit.circuit import Gate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import matplotlib.pyplot as plt
from U_config import *
from state_preparation import *

# ============================================================
# Hardware / simulator / statevector
#
# USE_REAL_HARDWARE
# If you can specify a device. If it is false you use the least busy one.
#
# USE_STATEVECTOR
# True: noiseless statevector simulation; no measurement sampling.
#       Overrides USE_REAL_HARDWARE and turns off all four noise flags below.
# False: original shot-based execution, with your hardware/noise settings.
# Both modes run ALL selected folded circuits and the same extrapolation.
# ============================================================
USE_REAL_HARDWARE = False
DEVICE_NAME = False
# DEVICE_NAME = "ibm_strasbourg"

USE_STATEVECTOR = True

shots = 10**3  # Used only when USE_STATEVECTOR is False.


# ============================================================
# Depth of the unitary U
# ============================================================

DEPTH = 6

# ============================================================
# Number of qubits in U
# ============================================================

NUM_U_QUBITS = 2

# ============================================================
# Circuit drawing
# If this flag is turn on, it draws only circuit and exit the code.
# ============================================================

DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 3


# ============================================================
# Number of foldings
# ============================================================

depth_folded_circuits = [1, 3, 5, 7]



# ============================================================
# Noise flags for a simulator

# Turn off all noise when drawing or using noiseless statevector simulation
# ============================================================

USE_DEPOLARIZING = True
USE_AMPLITUDE_DAMPING = False
USE_PHASE_DAMPING = False
USE_COHERENT_OVERROTATION = False

if USE_STATEVECTOR:
    USE_REAL_HARDWARE = False

if DRAW_CIRCUIT or USE_STATEVECTOR:
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

def unitary_initial_preparation(qc):
    """Keep initial preparation deterministic in no-shot mode.

    A standard initialize(...) on fresh |0> wires is supported: expand its
    definition and omit only resets on untouched |0> wires. No statevector
    or density matrix is constructed here. A reset of an already-used wire,
    measurement, classical feedback or channel in state preparation is
    rejected rather than silently sampling it. Ordinary unitary gates and
    composite gates are preserved.
    """
    out = qc.copy_empty_like()
    out.global_phase = qc.global_phase
    touched = set()

    def append_instruction(operation, wires, clbits):
        if clbits or getattr(operation, "condition", None) is not None:
            raise ValueError(
                "USE_STATEVECTOR requires pure, deterministic input preparation; "
                "measurements and classical feedback in state_preparation.py "
                "are not supported."
            )
        if operation.name in ("barrier", "delay"):
            out.append(operation, wires)
        elif operation.name == "reset":
            if any(wire in touched for wire in wires):
                raise ValueError(
                    "USE_STATEVECTOR cannot reset an already-used input wire. "
                    "Use unitary pure-state preparation instead."
                )
            # All circuit wires start in |0>; this reset has no effect.
        elif isinstance(operation, Gate):
            out.append(operation, wires)
            touched.update(wires)
        elif operation.name in ("measure", "if_else", "kraus", "superop", "quantum_channel"):
            raise ValueError(
                f"{operation.name!r} in input preparation is not supported "
                "in noiseless, no-shot statevector mode."
            )
        else:
            definition = operation.definition
            if definition is None:
                raise ValueError(
                    f"Cannot certify input instruction {operation.name!r} as "
                    "deterministic unitary preparation for USE_STATEVECTOR."
                )
            out.global_phase += definition.global_phase
            wire_map = dict(zip(definition.qubits, wires))
            for item in definition.data:
                append_instruction(
                    item.operation,
                    [wire_map[wire] for wire in item.qubits],
                    item.clbits,
                )

    for item in qc.data:
        append_instruction(item.operation, list(item.qubits), item.clbits)
    return out


def apply_coherent_correction(qc, qA, qB, target):
    """No-shot equivalent of the ORIGINAL apply_correction(), not a new rule.

    The original measures qA -> c_pair[1], qB -> c_pair[0]. Therefore
    00 -> XZ, 01 -> Z, 10 -> X, 11 -> I is Z if qA=0, THEN X if qB=0.
    X conjugation implements quantum controls on |0> rather than |1>.
    The Bell wires are retained and never reset or measured in this mode.
    """
    qc.x(qA)
    qc.cz(qA, target)
    qc.x(qA)

    qc.x(qB)
    qc.cx(qB, target)
    qc.x(qB)


# ============================================================
# Backend
# ============================================================


if USE_STATEVECTOR:
    backend = AerSimulator(
        method="statevector",
        precision="double",
        zero_threshold=0.0,
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

folded_circuits = []

if DRAW_CIRCUIT:
    depths_to_build = [DRAW_DEPTH_FOLDED]
else:
    depths_to_build = depth_folded_circuits


for depth_folded in depths_to_build:
    num_physical_qubits = depth_folded * NUM_U_QUBITS

    q = QuantumRegister(
        num_physical_qubits,
        'q'
    )

    qc = QuantumCircuit(q)

    num_fold = (depth_folded - 1) // 2

    # --------------------------------------------------------
    # Classical register for EACH Bell measurement
    # --------------------------------------------------------
    classical_pairs = []

    for fold in reversed(range(num_fold)):

        fold_pairs = []

        for k in range(NUM_U_QUBITS):
            c_pair = ClassicalRegister(
                2,
                f'c{fold}_{k}'
            )

            qc.add_register(c_pair)
            fold_pairs.append(c_pair)

        classical_pairs.append(fold_pairs)
    c_out = ClassicalRegister(NUM_U_QUBITS, "out")
    qc.add_register(c_out)
    # ========================================================
    # 1. Prepare input state
    # ========================================================
    input_block = get_block(q, 0)

    prepare_initial_state(
        qc,
        input_block
    )
    if USE_STATEVECTOR and not DRAW_CIRCUIT:
        qc = unitary_initial_preparation(qc)

    # ========================================================
    # 2. Prepare psi-minus states
    #
    # depth_folded = 3:
    #   (q1,q2)
    #
    # depth_folded = 5:
    #   (q1,q2), (q3,q4)
    #
    # etc.
    # ========================================================
    for fold in range(num_fold):

        left_block = get_block(q,2 * fold + 1)

        right_block = get_block(q,2 * fold + 2)

        for k in range(NUM_U_QUBITS):
            psi_minus(
                qc,
                [
                    left_block[k],
                    right_block[k]
                ]
            )

    qc.barrier()

    # ========================================================
    # 3. Apply U to every qubit
    # ========================================================
    for block_index in range(depth_folded):
        block = get_block(q,block_index)
        applyU(qc, block, block_index)
    qc.barrier()


    # ========================================================
    # 4. Bell measurement basis transformation
    #
    # depth_folded = 3:
    #   (q0,q1)
    #
    # depth_folded = 5:
    #   (q0,q1), (q2,q3)
    #
    # etc.
    # ========================================================
    for fold in range(num_fold):

        pair_block_1 = get_block(q, 2 * fold)
        pair_block_2 = get_block(q, 2 * fold + 1)

        for k in range(NUM_U_QUBITS):
            bell_measure(
                qc,
                [
                    pair_block_1[k],
                    pair_block_2[k]
                ]
            )

    qc.barrier()


    # ========================================================
    # 5. All Bell measurements
    # Do not append steps 4 and 5.
    # If we append them, the measurements will be performed sequentially.
    # ========================================================
    for fold in range(num_fold):

        pair_block_1 = get_block(q, 2 * fold)
        pair_block_2 = get_block(q, 2 * fold + 1)

        for k in range(NUM_U_QUBITS):
            qA = pair_block_1[k]
            qB = pair_block_2[k]

            c_pair = classical_pairs[fold][k]

            if not USE_STATEVECTOR or DRAW_CIRCUIT:
                qc.measure(qA, c_pair[1])
                qc.measure(qB, c_pair[0])
            # Statevector: defer BOTH Bell measurements. Their quantum wires
            # control the equivalent corrections below; no branch is sampled.
    # ========================================================
    # 6. Post-processing
    #
    # All corrections are applied to the LAST qubit.
    #
    # depth_folded = 1:
    #   no correction
    #
    # depth_folded = 3:
    #   correction from (q0,q1) -> q2
    #
    # depth_folded = 5:
    #   correction from (q0,q1) -> q4
    #   correction from (q2,q3) -> q4
    #
    # depth_folded = 7:
    #   correction from (q0,q1) -> q6
    #   correction from (q2,q3) -> q6
    #   correction from (q4,q5) -> q6
    #
    # ========================================================
    output_block = get_block(q,depth_folded - 1)

    for fold in range(num_fold):

        for k in range(NUM_U_QUBITS):
            if USE_STATEVECTOR and not DRAW_CIRCUIT:
                pair_block_1 = get_block(q, 2 * fold)
                pair_block_2 = get_block(q, 2 * fold + 1)
                apply_coherent_correction(
                    qc,
                    pair_block_1[k],  # qA: originally c_pair[1]
                    pair_block_2[k],  # qB: originally c_pair[0]
                    output_block[k],
                )
            else:
                apply_correction(
                    qc,
                    classical_pairs[fold][k],
                    output_block[k]
                )

    # ========================================================
    # 7. Final output measurement
    # ========================================================
    if USE_STATEVECTOR and not DRAW_CIRCUIT:
        # I on the Bell wires, Z on each FINAL output wire.
        # The joint pure state includes every Bell outcome coherently.
        # This returns their probability-weighted output expectation without
        # measurement sampling, branch selection or a density matrix.
        qc.save_expectation_value(
            Pauli("Z" * NUM_U_QUBITS),
            output_block,
            label="exact_expectation",
        )
    else:
        for k in range(NUM_U_QUBITS):
            qc.measure(
                output_block[k],
                c_out[k]
            )

    # ========================================================
    # Save circuit
    # ========================================================

    folded_circuits.append(qc)

    # ========================================================
    # Draw circuit only
    # ========================================================

    if DRAW_CIRCUIT:
        print(f"\nDepth_folded = {depth_folded}")

        qc.draw(
            'mpl',
            style='clifford'
        )

        plt.show()

        # Stop here. Do not proceed to transpilation/execution.
        raise SystemExit


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

if USE_STATEVECTOR:
    # All Bell measurements are coherently deferred; there are no measurements,
    job = backend.run(exec_circuits, shots=1)
    result = job.result()
    if not result.success:
        raise RuntimeError(f"Statevector simulation failed: {result.status}")

    expectation_values = []
    for i in range(len(folded_circuits)):
        expectation = float(np.real(result.data(i)["exact_expectation"]))
        if not np.isfinite(expectation):
            raise RuntimeError(f"Non-finite expectation for circuit {i}.")
        expectation_values.append(expectation)

else:
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

if USE_STATEVECTOR:
    print("Backend: AerSimulator (statevector)")

elif USE_REAL_HARDWARE:
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

print("\nExpectation values of global method:")
for fold_factor, expectation in zip(depths_to_build, expectation_values):
    print(f"fold factor {fold_factor}: {expectation:.12f}")


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

    zero_noise = np.polyval(
        coeffs,
        0.0
    )

    print(
        f"degree {degree}: "
        f"{zero_noise:.12f}"
    )
