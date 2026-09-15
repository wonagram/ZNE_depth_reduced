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

def apply_noise(qc, wire, axis, epsilon):

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
            error_circuit.rx(epsilon, 0)

        elif axis == "y":
            error_circuit.ry(epsilon, 0)

        elif axis == "z":
            error_circuit.rz(epsilon, 0)

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

def applyU(qc, wire):

    for axis, theta in U_GATES[:DEPTH]:

        apply_rotation(qc, wire, axis, theta)

        # Manual noise only for simulator
        if not USE_REAL_HARDWARE:
            apply_noise(
                qc,
                wire,
                axis,
                +overrotation_epsilon
            )


def applyUdagger(qc, wire):

    # Reverse the gates used in U and negate their angles
    for axis, theta in reversed(U_GATES[:DEPTH]):

        apply_rotation(qc, wire, axis, -theta)

        # Manual noise only for simulator
        if not USE_REAL_HARDWARE:
            apply_noise(
                qc,
                wire,
                axis,
                -overrotation_epsilon
            )


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
        dynamic_circuits=True # To make it consistent with depth-reduced method
    )

else:

    backend = AerSimulator()
    backend.set_options(seed_simulator=150)


# ============================================================
# Circuit running
# ============================================================

folded_circuits = []

for depth_folded in depth_folded_circuits:
    q = QuantumRegister(1, 'q')
    c = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(q, c)

    qc.ry(alpha, q[0])
    qc.rz(beta, q[0])
    qc.barrier()


    num_fold = (depth_folded - 1) // 2

    applyU(qc, q[0])
    for _ in range(num_fold):
        applyUdagger(qc,q[0])
        applyU(qc,q[0])

    qc.measure(q[0], c[0])
    folded_circuits.append(qc)

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

all_counts = [
    result[i].join_data().get_counts()
    for i in range(len(folded_circuits))
]

expectation_values = []

for counts in all_counts:

    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots

    expectation = p0 - p1

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


print(f"\nExpectation values of circuit_folded method:\n"
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
