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
import matplotlib.pyplot as plt


shots = 10**7
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



overrotation_x_circuit = QuantumCircuit(1)
overrotation_x_circuit.rx(overrotation_epsilon, 0)

overrotation_y_circuit = QuantumCircuit(1)
overrotation_y_circuit.ry(overrotation_epsilon, 0)

overrotation_z_circuit = QuantumCircuit(1)
overrotation_z_circuit.rz(overrotation_epsilon, 0)


coherent_x_noise = coherent_unitary_error(
    Operator(overrotation_x_circuit)
).to_instruction()

coherent_y_noise = coherent_unitary_error(
    Operator(overrotation_y_circuit)
).to_instruction()

coherent_z_noise = coherent_unitary_error(
    Operator(overrotation_z_circuit)
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

        if axis == "x":
            qc.append(
                coherent_x_noise,
                [wire]
            )

        elif axis == "y":
            qc.append(
                coherent_y_noise,
                [wire]
            )

        elif axis == "z":
            qc.append(
                coherent_z_noise,
                [wire]
            )


backend = AerSimulator()
backend.set_options(seed_simulator=150)


def psi_minus(qc, wires):
    qc.h(wires[0])
    qc.cx(wires[0], wires[1])
    qc.z(wires[0])
    qc.x(wires[1])

def bell_measure(qc, wires):
    qc.cx(wires[0], wires[1])
    qc.h(wires[0])

def applyU(qc, wires):

    for wire in wires:
        qc.rx(theta_x1, wire)
        apply_noise(qc, wire, "x")

        qc.ry(theta_y1, wire)
        apply_noise(qc, wire, "y")

        qc.rz(theta_z1, wire)
        apply_noise(qc, wire, "z")

        qc.rx(theta_x2, wire)
        apply_noise(qc, wire, "x")

        qc.ry(theta_y2, wire)
        apply_noise(qc, wire, "y")

        qc.rz(theta_z2, wire)
        apply_noise(qc, wire, "z")

        qc.rx(theta_x3, wire)
        apply_noise(qc, wire, "x")

        qc.ry(theta_y3, wire)
        apply_noise(qc, wire, "y")

        qc.rz(theta_z3, wire)
        apply_noise(qc, wire, "z")

        qc.rx(theta_x4, wire)
        apply_noise(qc, wire, "x")

        qc.ry(theta_y4, wire)
        apply_noise(qc, wire, "y")

        qc.rz(theta_z4, wire)
        apply_noise(qc, wire, "z")

        qc.rx(theta_x5, wire)
        apply_noise(qc, wire, "x")

        qc.ry(theta_y5, wire)
        apply_noise(qc, wire, "y")

        qc.rz(theta_z5, wire)
        apply_noise(qc, wire, "z")

        qc.rx(theta_x6, wire)
        apply_noise(qc, wire, "x")

        qc.ry(theta_y6, wire)
        apply_noise(qc, wire, "y")

        qc.rz(theta_z6, wire)
        apply_noise(qc, wire, "z")


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

#scale_factors = [1, 3, 5, 7, 9, 11]
scale_factors = [1]

for scale in scale_factors:
    q = QuantumRegister(scale, 'q')
    qc = QuantumCircuit(q)

    num_pairs = (scale - 1) // 2

    # --------------------------------------------------------
    # Classical register for EACH Bell measurement
    # --------------------------------------------------------
    classical_pairs = []
    for pair in range(num_pairs):
        c_pair = ClassicalRegister(2, f'c{pair}')
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
    # scale = 3:
    #   (q1,q2)
    #
    # scale = 5:
    #   (q1,q2), (q3,q4)
    #
    # etc.
    # ========================================================
    for pair in range(num_pairs):
        psi_minus(qc,[q[2*pair+1],q[2*pair+2]])
    qc.barrier()


    # ========================================================
    # 3. Apply U to every qubit
    # ========================================================
    applyU(qc,q)
    qc.barrier()


    # ========================================================
    # 4. Bell measurement basis transformation
    #
    # scale = 3:
    #   (q0,q1)
    #
    # scale = 5:
    #   (q0,q1), (q2,q3)
    #
    # etc.
    # ========================================================
    for pair in range(num_pairs):
        qA = q[2 * pair]
        qB = q[2 * pair + 1]
        bell_measure(qc, [qA, qB])
    qc.barrier()


    # ========================================================
    # 5. All Bell measurements
    # ========================================================
    for pair in range(num_pairs):
        qA = q[2 * pair]
        qB = q[2 * pair + 1]
        c_pair = classical_pairs[pair]
        qc.measure(qA, c_pair[1])
        qc.measure(qB, c_pair[0])
    qc.barrier()

    # ========================================================
    # 6. Post-processing
    #
    # All corrections are applied to the LAST qubit.
    #
    # scale = 1:
    #   no correction
    #
    # scale = 3:
    #   correction from (q0,q1) -> q2
    #
    # scale = 5:
    #   correction from (q0,q1) -> q4
    #   correction from (q2,q3) -> q4
    #
    # scale = 7:
    #   correction from (q0,q1) -> q6
    #   correction from (q2,q3) -> q6
    #   correction from (q4,q5) -> q6
    #
    # ========================================================
    target = q[scale - 1]

    for pair in range(num_pairs):
        apply_correction(
            qc,
            classical_pairs[pair],
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

    #print(f"\nScale factor = {scale}")
    #qc.draw('mpl')
    #plt.show()


# ============================================================
# Zero-noise extrapolation
# ============================================================
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

print(f"\nExpectation values of depth_reduced method:\n"
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
