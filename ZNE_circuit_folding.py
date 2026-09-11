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
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

shots = 10**4
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
# Noise flags
# ============================================================

USE_DEPOLARIZING = False
USE_AMPLITUDE_DAMPING = False
USE_PHASE_DAMPING = False
USE_COHERENT_OVERROTATION = False

# ============================================================
# Noise parameters
# ============================================================

depolarizing_strength = 0.02

amplitude_damping_strength = 0.02

phase_damping_strength = 0.02

# coherent over-rotation angle
overrotation_epsilon = 0.02


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
def applyU(qc, wire):

    qc.rx(theta_x1, wire)
    apply_noise(qc, wire, "x", +overrotation_epsilon)

    qc.ry(theta_y1, wire)
    apply_noise(qc, wire, "y", +overrotation_epsilon)

    qc.rz(theta_z1, wire)
    apply_noise(qc, wire, "z", +overrotation_epsilon)

    qc.rx(theta_x2, wire)
    apply_noise(qc, wire, "x", +overrotation_epsilon)

    qc.ry(theta_y2, wire)
    apply_noise(qc, wire, "y", +overrotation_epsilon)

    qc.rz(theta_z2, wire)
    apply_noise(qc, wire, "z", +overrotation_epsilon)

    qc.rx(theta_x3, wire)
    apply_noise(qc, wire, "x", +overrotation_epsilon)

    qc.ry(theta_y3, wire)
    apply_noise(qc, wire, "y", +overrotation_epsilon)

    qc.rz(theta_z3, wire)
    apply_noise(qc, wire, "z", +overrotation_epsilon)

    qc.rx(theta_x4, wire)
    apply_noise(qc, wire, "x", +overrotation_epsilon)

    qc.ry(theta_y4, wire)
    apply_noise(qc, wire, "y", +overrotation_epsilon)

    qc.rz(theta_z4, wire)
    apply_noise(qc, wire, "z", +overrotation_epsilon)

    qc.rx(theta_x5, wire)
    apply_noise(qc, wire, "x", +overrotation_epsilon)

    qc.ry(theta_y5, wire)
    apply_noise(qc, wire, "y", +overrotation_epsilon)

    qc.rz(theta_z5, wire)
    apply_noise(qc, wire, "z", +overrotation_epsilon)

    qc.rx(theta_x6, wire)
    apply_noise(qc, wire, "x", +overrotation_epsilon)

    qc.ry(theta_y6, wire)
    apply_noise(qc, wire, "y", +overrotation_epsilon)

    qc.rz(theta_z6, wire)
    apply_noise(qc, wire, "z", +overrotation_epsilon)


def applyUdagger(qc, wire):

    qc.rz(-theta_z6, wire)
    apply_noise(qc, wire, "z", -overrotation_epsilon)

    qc.ry(-theta_y6, wire)
    apply_noise(qc, wire, "y", -overrotation_epsilon)

    qc.rx(-theta_x6, wire)
    apply_noise(qc, wire, "x", -overrotation_epsilon)


    qc.rz(-theta_z5, wire)
    apply_noise(qc, wire, "z", -overrotation_epsilon)

    qc.ry(-theta_y5, wire)
    apply_noise(qc, wire, "y", -overrotation_epsilon)

    qc.rx(-theta_x5, wire)
    apply_noise(qc, wire, "x", -overrotation_epsilon)

    qc.rz(-theta_z4, wire)
    apply_noise(qc, wire, "z", -overrotation_epsilon)

    qc.ry(-theta_y4, wire)
    apply_noise(qc, wire, "y", -overrotation_epsilon)

    qc.rx(-theta_x4, wire)
    apply_noise(qc, wire, "x", -overrotation_epsilon)

    qc.rz(-theta_z3, wire)
    apply_noise(qc, wire, "z", -overrotation_epsilon)

    qc.ry(-theta_y3, wire)
    apply_noise(qc, wire, "y", -overrotation_epsilon)

    qc.rx(-theta_x3, wire)
    apply_noise(qc, wire, "x", -overrotation_epsilon)

    qc.rz(-theta_z2, wire)
    apply_noise(qc, wire, "z", -overrotation_epsilon)

    qc.ry(-theta_y2, wire)
    apply_noise(qc, wire, "y", -overrotation_epsilon)

    qc.rx(-theta_x2, wire)
    apply_noise(qc, wire, "x", -overrotation_epsilon)

    qc.rz(-theta_z1, wire)
    apply_noise(qc, wire, "z", -overrotation_epsilon)

    qc.ry(-theta_y1, wire)
    apply_noise(qc, wire, "y", -overrotation_epsilon)

    qc.rx(-theta_x1, wire)
    apply_noise(qc, wire, "x", -overrotation_epsilon)

backend = AerSimulator()
backend.set_options(seed_simulator=150)


folded_circuits = []

scale_factors = [1, 3, 5, 7, 9, 11]

for scale in scale_factors:
    q = QuantumRegister(1, 'q')
    c = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(q, c)

    qc.ry(alpha, q[0])
    qc.rz(beta, q[0])
    qc.barrier()


    num_pairs = (scale - 1) // 2

    applyU(qc, q[0])
    for _ in range(num_pairs):
        applyUdagger(qc,q[0])
        applyU(qc,q[0])

    qc.measure(q[0], c[0])
    folded_circuits.append(qc)

pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=0
)

exec_circuits = [
    pm.run(circuit)
    for circuit in folded_circuits
]

sampler = Sampler(backend)

job = sampler.run(
    exec_circuits,
    shots=shots
)

all_counts = [
    job.result()[i].join_data().get_counts()
    for i in range(len(folded_circuits))
]

expectation_values = []

for counts in all_counts:

    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots

    expectation = p0 - p1

    expectation_values.append(expectation)

print(f"\nExpectation values of circuit_folded method:\n"
      f"{[round(x, 5) for x in expectation_values]}")
# ============================================================
# Zero-noise extrapolation
# ============================================================
scale_factors_np = np.array(
    scale_factors,
    dtype=float
)

expectation_values_np = np.array(
    expectation_values,
    dtype=float
)


for degree in [1, 2, 3, 4]:
    coeffs = np.polyfit(
        scale_factors_np,
        expectation_values_np,
        deg=degree
    )

    zero_noise = np.polyval(coeffs, 0.0)

    print(
        f"degree {degree}: "
        f"{zero_noise:.6f}"
    )
expectation_values_np = np.array(
    expectation_values,
    dtype=float
)