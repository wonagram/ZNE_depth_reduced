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

DEPTH = 3

# ============================================================
# Number of qubits in U
# ============================================================

NUM_U_QUBITS = 1

# ============================================================
# Circuit drawing
# ============================================================

DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 3

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

def apply_noise(qc, wires, gate_type, epsilon):

    # ========================================================
    # 1-qubit rotation noise
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
                error_circuit.rx(epsilon, 0)

            elif gate_type == "y":
                error_circuit.ry(epsilon, 0)

            elif gate_type == "z":
                error_circuit.rz(epsilon, 0)

            coherent_noise = coherent_unitary_error(
                Operator(error_circuit)
            ).to_instruction()

            qc.append(
                coherent_noise,
                [wire]
            )


    # ========================================================
    # CNOT noise
    # ========================================================

    elif gate_type == "cx":

        control = wires[0]
        target = wires[1]

        if USE_DEPOLARIZING:
            qc.append(
                depolarizing_noise_2q,
                [control, target]
            )

        if USE_AMPLITUDE_DAMPING:
            qc.append(
                amplitude_damping_noise,
                [control]
            )
            qc.append(
                amplitude_damping_noise,
                [target]
            )

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

def applyU(qc, wires):

    U_layers = U_GATES[NUM_U_QUBITS]

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
                        gate_type,
                        +overrotation_epsilon
                    )

            # ====================================================
            # CNOT
            # ====================================================

            elif gate_type == "cx":

                _, control, target = gate

                control_wire = wires[control]
                target_wire = wires[target]

                qc.cx(
                    control_wire,
                    target_wire
                )

                if not USE_REAL_HARDWARE:
                    apply_noise(
                        qc,
                        [control_wire, target_wire],
                        "cx",
                        0.0
                    )

            else:
                raise ValueError(
                    f"Unknown gate type: {gate_type}"
                )


def applyUdagger(qc, wires):

    U_layers = U_GATES[NUM_U_QUBITS]

    # Reverse all layers
    for layer in reversed(U_layers[:DEPTH]):

        # Reverse gates inside each layer as well
        for gate in reversed(layer):

            gate_type = gate[0]

            # ====================================================
            # Single-qubit rotation dagger
            # ====================================================

            if gate_type in ["x", "y", "z"]:

                _, local_qubit, theta = gate

                wire = wires[local_qubit]

                apply_rotation(
                    qc,
                    wire,
                    gate_type,
                    -theta
                )

                if not USE_REAL_HARDWARE:
                    apply_noise(
                        qc,
                        [wire],
                        gate_type,
                        -overrotation_epsilon
                    )

            # ====================================================
            # CNOT dagger = CNOT
            # ====================================================

            elif gate_type == "cx":

                _, control, target = gate

                control_wire = wires[control]
                target_wire = wires[target]

                qc.cx(
                    control_wire,
                    target_wire
                )

                if not USE_REAL_HARDWARE:
                    apply_noise(
                        qc,
                        [control_wire, target_wire],
                        "cx",
                        0.0
                    )

            else:
                raise ValueError(
                    f"Unknown gate type: {gate_type}"
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

    if DEVICE_NAME:
        backend = service.backend(DEVICE_NAME)

    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            dynamic_circuits=True,  # To make it consistent with depth-reduced method
        )

else:

    backend = AerSimulator()
    backend.set_options(seed_simulator=150)


# ============================================================
# Circuit running
# ============================================================

folded_circuits = []

if DRAW_CIRCUIT:
    depths_to_build = [DRAW_DEPTH_FOLDED]
else:
    depths_to_build = depth_folded_circuits

for depth_folded in depths_to_build:
    q = QuantumRegister(NUM_U_QUBITS, 'q')
    c = ClassicalRegister(NUM_U_QUBITS, 'c')
    qc = QuantumCircuit(q, c)

    # ========================================================
    # 1. Prepare initial n-qubit state
    # ========================================================

    prepare_initial_state(
        qc,
        q
    )

    qc.barrier()

    # ========================================================
    # 2. Circuit folding
    #
    # depth_folded = 1:
    #     U
    #
    # depth_folded = 3:
    #     U -> Udagger -> U
    #
    # depth_folded = 5:
    #     U -> Udagger -> U -> Udagger -> U
    # ========================================================

    num_fold = (depth_folded - 1) // 2

    applyU(
        qc,
        q
    )

    for _ in range(num_fold):

        applyUdagger(
            qc,
            q
        )

        applyU(
            qc,
            q
        )


    # ========================================================
    # 3. Final measurement
    # ========================================================

    for k in range(NUM_U_QUBITS):

        qc.measure(
            q[k],
            c[k]
        )

    folded_circuits.append(qc)

    # ========================================================
    # Draw circuit only
    # ========================================================

    if DRAW_CIRCUIT:

        print(
            f"\nDepth_folded = {depth_folded}"
        )

        qc.draw(
            'mpl',
            style='clifford'
        )

        plt.show()

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
# <Z^{\otimes n}> = sum_x (-1)^{|x|} P(x)
# ============================================================

expectation_values = []

for i in range(len(folded_circuits)):

    counts = result[i].join_data().get_counts()

    total_counts = sum(
        counts.values()
    )

    expectation = 0.0

    for bitstring, count in counts.items():

        bitstring = bitstring.replace(
            " ",
            ""
        )

        eigenvalue = (
            (-1) ** bitstring.count("1")
        )

        expectation += (
            eigenvalue
            * count
            / total_counts
        )

    expectation_values.append(
        expectation
    )


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
