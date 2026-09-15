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


# ============================================================
# Hardware / simulator
# ============================================================

USE_REAL_HARDWARE = False
shots = 10**4

# ============================================================
# Depth of the unitary U
# ============================================================

DEPTH = 2

# ============================================================
# Number of foldings
# ============================================================

depth_folded_circuits = [1, 3, 5, 7, 9, 11]

# ============================================================
# Noise flags for a simulator
# ============================================================

USE_DEPOLARIZING = True
USE_AMPLITUDE_DAMPING = False
USE_PHASE_DAMPING = True
USE_COHERENT_OVERROTATION = True

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

alpha   = 0.7
beta    = 0.4

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

U_GATES = [
    ("x", theta_x1),
    ("y", theta_y1),
    ("z", theta_z1),

    ("x", theta_x2),
    ("y", theta_y2),
    ("z", theta_z2),

    ("x", theta_x3),
    ("y", theta_y3),
    ("z", theta_z3),

    ("x", theta_x4),
    ("y", theta_y4),
    ("z", theta_z4),

    ("x", theta_x5),
    ("y", theta_y5),
    ("z", theta_z5),

    ("x", theta_x6),
    ("y", theta_y6),
    ("z", theta_z6),
]

if not 1 <= DEPTH <= len(U_GATES):
    raise ValueError(
        f"DEPTH must be between 1 and {len(U_GATES)}, got {DEPTH}."
    )

# ============================================================
# Noise instructions
# ============================================================

depolarizing_noise = depolarizing_error(
    depolarizing_strength,
    1
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

def apply_noise(qc, wire, axis):

    if USE_DEPOLARIZING:
        qc.append(
            depolarizing_noise,
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

        if axis == "x":
            error_circuit.rx(overrotation_epsilon, 0)

        elif axis == "y":
            error_circuit.ry(overrotation_epsilon, 0)

        elif axis == "z":
            error_circuit.rz(overrotation_epsilon, 0)

        coherent_noise = coherent_unitary_error(
            Operator(error_circuit)
        ).to_instruction()

        qc.append(
            coherent_noise,
            [wire]
        )


def apply_rotation(qc, wire, axis, angle):

    if axis == "x":
        qc.rx(angle, wire)

    elif axis == "y":
        qc.ry(angle, wire)

    elif axis == "z":
        qc.rz(angle, wire)


def applyU(qc, wires):

    for wire in wires:

        for axis, theta in U_GATES[:DEPTH]:

            apply_rotation(
                qc,
                wire,
                axis,
                theta
            )

            # Artificial noise only for simulator
            if not USE_REAL_HARDWARE:
                apply_noise(
                    qc,
                    wire,
                    axis,
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

    backend = service.least_busy(
        operational=True,
        simulator=False,
        dynamic_circuits=True
    )
else:

    backend = AerSimulator()
    backend.set_options(seed_simulator=150)

folded_circuits = []

for depth_folded in depth_folded_circuits:
    q = QuantumRegister(depth_folded, 'q')
    qc = QuantumCircuit(q)

    num_fold = (depth_folded - 1) // 2

    # --------------------------------------------------------
    # Classical register for EACH Bell measurement
    # --------------------------------------------------------
    classical_pairs = []
    for fold in range(num_fold):
        c_pair = ClassicalRegister(2, f'c{fold}')
        qc.add_register(c_pair)
        classical_pairs.append(c_pair)
    c_out = ClassicalRegister(1, "out")
    qc.add_register(c_out)

    # ========================================================
    # 1. Prepare input state
    # ========================================================
    qc.ry(alpha, q[0])
    qc.rz(beta, q[0])


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
        psi_minus(qc,[q[2*fold+1],q[2*fold+2]])
    qc.barrier()


    # ========================================================
    # 3. Apply U to every qubit
    # ========================================================
    applyU(qc,q)
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
        qA = q[2 * fold]
        qB = q[2 * fold + 1]
        bell_measure(qc, [qA, qB])
    qc.barrier()


    # ========================================================
    # 5. All Bell measurements
    # ========================================================
    for fold in range(num_fold):
        qA = q[2 * fold]
        qB = q[2 * fold + 1]
        c_pair = classical_pairs[fold]
        qc.measure(qA, c_pair[1])
        qc.measure(qB, c_pair[0])
    qc.barrier()

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
    target = q[depth_folded - 1]

    for fold in range(num_fold):
        apply_correction(
            qc,
            classical_pairs[fold],
            target
        )

    # ========================================================
    # 7. Final output measurement
    # ========================================================
    qc.measure(target, c_out[0])



    # ========================================================
    # Save circuit
    # ========================================================

    folded_circuits.append(qc)

    #print(f"\n Depth_folded = {depth_folded}")
    #qc.draw('mpl')
    #plt.show()


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
# Extract final output-qubit probability
# ============================================================
expectation_values = []
for i in range(len(folded_circuits)):

    out_counts = result[i].data.out.get_counts()

    p0 = out_counts.get("0", 0) / shots
    p1 = out_counts.get("1", 0) / shots

    expectation_values.append(p0-p1)
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
