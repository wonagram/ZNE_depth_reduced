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

# ============================================================
# Hardware / simulator

# If you can specify a device. If it is false you use the least busy one.
# ============================================================

USE_REAL_HARDWARE = False
DEVICE_NAME = False
# DEVICE_NAME = "ibm_strasbourg"

shots = 10**4

# ============================================================
# Depth of the unitary U
# ============================================================

DEPTH = 5


# ============================================================
# Number of qubits in U
# ============================================================

NUM_U_QUBITS = 1


# ============================================================
# Circuit drawing
# If this flag is turn on, it draws only circuit and exit the code.
# ============================================================

DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 5


# ============================================================
# Number of foldings
# ============================================================

depth_folded_circuits = [1, 3, 5, 7, 9, 11]



# ============================================================
# Noise flags for a simulator

# Turn off all noise when drawing circuit
# ============================================================

USE_DEPOLARIZING = True
USE_AMPLITUDE_DAMPING = False
USE_PHASE_DAMPING = True
USE_COHERENT_OVERROTATION = True

if DRAW_CIRCUIT:
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


# ============================================================
# Initial state preparation
# ============================================================

alpha_0   = 0.7
alpha_1   = 0.6
alpha_2   = 0.5
alpha_3   = 0.4

beta_0    = 0.4
beta_1    = 0.3
beta_2    = 0.2
beta_3    = 0.1

# ============================================================
# U parameters
# ============================================================
theta_x1 = 0.5
theta_y1 = 0.6
theta_z1 = 0.8
theta_x2 = 0.4
theta_y2 = 0.4
theta_z2 = 0.4
theta_x3 = 0.6
theta_y3 = 0.5
theta_z3 = 0.9
theta_x4 = 0.3
theta_y4 = 0.4
theta_z4 = 0.5
theta_x5 = 0.8
theta_y5 = 0.7
theta_z5 = 0.6
theta_x6 = 0.7
theta_y6 = 0.7
theta_z6 = 0.7

# ============================================================
# U gate sequence
# ============================================================
U_GATES = {
    1: [
        # Layer 1
        [("x", 0, theta_x1)],

        # Layer 2
        [("y", 0, theta_y1)],

        # Layer 3
        [("z", 0, theta_z1)],

        # Layer 4
        [("x", 0, theta_x2)],

        # Layer 5
        [("y", 0, theta_y2)],

        # Layer 6
        [("z", 0, theta_z2)],

        # Layer 7
        [("x", 0, theta_x3)],

        # Layer 8
        [("y", 0, theta_y3)],

        # Layer 9
        [("z", 0, theta_z3)],

        # Layer 10
        [("x", 0, theta_x4)],

        # Layer 11
        [("y", 0, theta_y4)],

        # Layer 12
        [("z", 0, theta_z4)],

        # Layer 13
        [("x", 0, theta_x5)],

        # Layer 14
        [("y", 0, theta_y5)],

        # Layer 15
        [("z", 0, theta_z5)],

        # Layer 16
        [("x", 0, theta_x6)],

        # Layer 17
        [("y", 0, theta_y6)],

        # Layer 18
        [("z", 0, theta_z6)],
    ],

    2: [
        # Layer 1
        [
            ("x", 0, theta_x1),
            ("x", 1, theta_x1),
        ],

        # Layer 2
        [
            ("cx", 0, 1),
        ],

        # Layer 3
        [
            ("y", 0, theta_y1),
            ("y", 1, theta_y1),
        ],

        # Layer 4
        [
            ("cx", 1, 0),
        ],

        # Layer 5
        [
            ("z", 0, theta_z1),
            ("z", 1, theta_z1),
        ],

        # Layer 6
        [
            ("cx", 0, 1),
        ],

        # Layer 7
        [
            ("x", 0, theta_x2),
            ("x", 1, theta_x2),
        ],

        # Layer 8
        [
            ("cx", 1, 0),
        ],

        # Layer 9
        [
            ("y", 0, theta_y2),
            ("y", 1, theta_y2),
        ],

        # Layer 10
        [
            ("cx", 0, 1),
        ],

        # Layer 11
        [
            ("z", 0, theta_z2),
            ("z", 1, theta_z2),
        ],

        # Layer 12
        [
            ("cx", 1, 0),
        ],

        # Layer 13
        [
            ("x", 0, theta_x3),
            ("x", 1, theta_x3),
        ],

        # Layer 14
        [
            ("cx", 0, 1),
        ],

        # Layer 15
        [
            ("y", 0, theta_y3),
            ("y", 1, theta_y3),
        ],

        # Layer 16
        [
            ("cx", 1, 0),
        ],

        # Layer 17
        [
            ("z", 0, theta_z3),
            ("z", 1, theta_z3),
        ],

        # Layer 18
        [
            ("cx", 0, 1),
        ],
    ],

    3: [
        # Layer 1
        [
            ("x", 0, theta_x1),
            ("x", 1, theta_x1),
            ("x", 2, theta_x1),
        ],

        # Layer 2
        [
            ("cx", 0, 1),
        ],

        # Layer 3
        [
            ("cx", 1, 2),
        ],

        # Layer 4
        [
            ("y", 0, theta_y1),
            ("y", 1, theta_y1),
            ("y", 2, theta_y1),
        ],

        # Layer 5
        [
            ("cx", 2, 1),
        ],

        # Layer 6
        [
            ("cx", 0, 1),
        ],

        # Layer 7
        [
            ("z", 0, theta_z1),
            ("z", 1, theta_z1),
            ("z", 2, theta_z1),
        ],

        # Layer 8
        [
            ("cx", 1, 2),
        ],

        # Layer 9
        [
            ("cx", 2, 1),
        ],

        # Layer 10
        [
            ("x", 0, theta_x2),
            ("x", 1, theta_x2),
            ("x", 2, theta_x2),
        ],

        # Layer 11
        [
            ("cx", 0, 1),
        ],

        # Layer 12
        [
            ("cx", 1, 2),
        ],

        # Layer 13
        [
            ("y", 0, theta_y2),
            ("y", 1, theta_y2),
            ("y", 2, theta_y2),
        ],

        # Layer 14
        [
            ("cx", 2, 1),
        ],

        # Layer 15
        [
            ("cx", 0, 1),
        ],

        # Layer 16
        [
            ("z", 0, theta_z2),
            ("z", 1, theta_z2),
            ("z", 2, theta_z2),
        ],

        # Layer 17
        [
            ("cx", 1, 2),
        ],

        # Layer 18
        [
            ("cx", 2, 1),
        ],
    ],

    4: [
        # Layer 1
        [
            ("x", 0, theta_x1),
            ("x", 1, theta_x1),
            ("x", 2, theta_x1),
            ("x", 3, theta_x1),
        ],

        # Layer 2
        [
            ("cx", 0, 1),
            ("cx", 2, 3),
        ],

        # Layer 3
        [
            ("y", 0, theta_y1),
            ("y", 1, theta_y1),
            ("y", 2, theta_y1),
            ("y", 3, theta_y1),
        ],

        # Layer 4
        [
            ("cx", 1, 2),
        ],

        # Layer 5
        [
            ("z", 0, theta_z1),
            ("z", 1, theta_z1),
            ("z", 2, theta_z1),
            ("z", 3, theta_z1),
        ],

        # Layer 6
        [
            ("cx", 3, 2),
            ("cx", 1, 0),
        ],

        # Layer 7
        [
            ("x", 0, theta_x2),
            ("x", 1, theta_x2),
            ("x", 2, theta_x2),
            ("x", 3, theta_x2),
        ],

        # Layer 8
        [
            ("cx", 2, 1),
        ],

        # Layer 9
        [
            ("y", 0, theta_y2),
            ("y", 1, theta_y2),
            ("y", 2, theta_y2),
            ("y", 3, theta_y2),
        ],

        # Layer 10
        [
            ("cx", 0, 1),
            ("cx", 2, 3),
        ],

        # Layer 11
        [
            ("z", 0, theta_z2),
            ("z", 1, theta_z2),
            ("z", 2, theta_z2),
            ("z", 3, theta_z2),
        ],

        # Layer 12
        [
            ("cx", 1, 2),
        ],

        # Layer 13
        [
            ("x", 0, theta_x3),
            ("x", 1, theta_x3),
            ("x", 2, theta_x3),
            ("x", 3, theta_x3),
        ],

        # Layer 14
        [
            ("cx", 3, 2),
            ("cx", 1, 0),
        ],

        # Layer 15
        [
            ("y", 0, theta_y3),
            ("y", 1, theta_y3),
            ("y", 2, theta_y3),
            ("y", 3, theta_y3),
        ],

        # Layer 16
        [
            ("cx", 2, 1),
        ],
    ],
}

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
# State preparation
# ============================================================

def prepare_initial_state(qc, wires):

    if NUM_U_QUBITS == 1:

        qc.ry(alpha_0, wires[0])
        qc.rz(beta_0, wires[0])


    elif NUM_U_QUBITS == 2:

        qc.ry(alpha_0, wires[0])
        qc.ry(alpha_1, wires[1])

        qc.rz(beta_0, wires[0])
        qc.rz(beta_1, wires[1])

        qc.cx(wires[0], wires[1])


    elif NUM_U_QUBITS == 3:

        qc.ry(alpha_0, wires[0])
        qc.ry(alpha_1, wires[1])
        qc.ry(alpha_2, wires[2])

        qc.rz(beta_0, wires[0])
        qc.rz(beta_1, wires[1])
        qc.rz(beta_2, wires[2])

        qc.cx(wires[0], wires[1])
        qc.cx(wires[1], wires[2])


    elif NUM_U_QUBITS == 4:

        qc.ry(alpha_0, wires[0])
        qc.ry(alpha_1, wires[1])
        qc.ry(alpha_2, wires[2])
        qc.ry(alpha_3, wires[3])

        qc.rz(beta_0, wires[0])
        qc.rz(beta_1, wires[1])
        qc.rz(beta_2, wires[2])
        qc.rz(beta_3, wires[3])

        qc.cx(wires[0], wires[1])
        qc.cx(wires[1], wires[2])
        qc.cx(wires[2], wires[3])

    else:
        raise ValueError(
            f"Unsupported NUM_U_QUBITS = {NUM_U_QUBITS}"
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
    amplitude_damping_strength
).to_instruction()

phase_damping_noise = phase_damping_error(
    phase_damping_strength
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

        if USE_DEPOLARIZING:
            qc.append(
                depolarizing_noise_2q,
                [control, target]
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

    for fold in range(num_fold):

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

            qc.measure(qA, c_pair[1])
            qc.measure(qB, c_pair[0])
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
            apply_correction(
                qc,
                classical_pairs[fold][k],
                output_block[k]
            )

    # ========================================================
    # 7. Final output measurement
    # ========================================================
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

print(f"\nExpectation values of depth_reduced method:\n"
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


for degree in [1, 2, 3, 4]:
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
