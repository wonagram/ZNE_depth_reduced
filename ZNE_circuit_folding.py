from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import matplotlib.pyplot as plt

from U_config import (
    get_unitary_max_depth,
    normalize_unitary_type,
    select_unitary_components,
    selected_unitary_summary,
)
from unitary_tools import NoiseController, append_unitary_block
from state_preparation import *

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

# ============================================================
# Circuit drawing
# ============================================================
DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 5

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

if DRAW_CIRCUIT or USE_STATEVECTOR:
    USE_DEPOLARIZING = False
    USE_AMPLITUDE_DAMPING = False
    USE_PHASE_DAMPING = False
    USE_COHERENT_OVERROTATION = False

# ============================================================
# Select U.  DEPTH may cut an RXX/RYY/RZZ component after slice 1 or 2.
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

# ============================================================
# Backend
# ============================================================


if USE_STATEVECTOR:

    backend = AerSimulator(
        method="statevector",
        precision="double",
        zero_threshold=0.0,
        max_parallel_experiments=1,
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
            dynamic_circuits=True,  # To make it consistent with depth-reduced method
        )

else:

    backend = AerSimulator()
    backend.set_options(seed_simulator=150)


# ============================================================
# Circuit running
# ============================================================

folded_circuits = []
COMPILATION_TOTALS = {
    "global_y_pairs": 0,
    "h_pairs": 0,
    "basis_rx_pairs": 0,
}

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

    compile_stats = append_unitary_block(
        qc,
        q,
        SELECTED_COMPONENTS,
        inverse=False,
        dress_cx=False,
        noise_controller=NOISE_CONTROLLER,
        use_noise=(
            not USE_REAL_HARDWARE
            and not DRAW_CIRCUIT
            and not USE_STATEVECTOR
        ),
        compile_cancellations=True,
    )
    for key, value in compile_stats.items():
        COMPILATION_TOTALS[key] += value

    for _ in range(num_fold):
        compile_stats = append_unitary_block(
            qc,
            q,
            SELECTED_COMPONENTS,
            inverse=True,
            dress_cx=False,
            noise_controller=NOISE_CONTROLLER,
            use_noise=(
                not USE_REAL_HARDWARE
                and not DRAW_CIRCUIT
                and not USE_STATEVECTOR
            ),
            compile_cancellations=True,
        )
        for key, value in compile_stats.items():
            COMPILATION_TOTALS[key] += value

        compile_stats = append_unitary_block(
            qc,
            q,
            SELECTED_COMPONENTS,
            inverse=False,
            dress_cx=False,
            noise_controller=NOISE_CONTROLLER,
            use_noise=(
                not USE_REAL_HARDWARE
                and not DRAW_CIRCUIT
                and not USE_STATEVECTOR
            ),
            compile_cancellations=True,
        )
        for key, value in compile_stats.items():
            COMPILATION_TOTALS[key] += value


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
# Select readout for execution
# ============================================================
if USE_STATEVECTOR:
    simulation_circuits = []
    observable = Pauli("Z" * NUM_U_QUBITS)

    for circuit in folded_circuits:
        qc_exact = circuit.remove_final_measurements(inplace=False)
        if any(item.operation.name == "measure" for item in qc_exact.data):
            raise ValueError(
                "USE_STATEVECTOR requires a pure-state preparation without "
                "intermediate measurements."
            )
        qc_exact.save_expectation_value(
            observable,
            list(range(NUM_U_QUBITS)),
            label="expectation_value",
        )
        simulation_circuits.append(qc_exact)
else:
    simulation_circuits = folded_circuits

# ============================================================
# Transpilation
# ============================================================
pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=0
)

exec_circuits = [
    pm.run(circuit)
    for circuit in simulation_circuits
]

# ============================================================
# Run every folded circuit and collect expectations
# ============================================================
if USE_STATEVECTOR:
    # One deterministic state evolution per circuit. There are no noise
    job = backend.run(exec_circuits, shots=1)
    result = job.result()
    if not result.success:
        raise RuntimeError(f"Statevector simulation failed: {result.status}")

    expectation_values = [
        float(result.data(i)["expectation_value"])
        for i in range(len(exec_circuits))
    ]
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

print(f"\nUNITARY_TYPE = {UNITARY_TYPE}")
print(selected_unitary_summary(UNITARY_TYPE, NUM_U_QUBITS, DEPTH))
print(f"DEPTH = {DEPTH}; maximum configured depth = {MAX_U_DEPTH}")
print(
    "One-pass U compilation removed pairs: "
    f"H={COMPILATION_TOTALS['h_pairs']}, "
    f"inverse RX(pi/2)={COMPILATION_TOTALS['basis_rx_pairs']}"
)

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
print(f"\nExpectation values of circuit_folded method:")
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

    zero_noise = np.polyval(
        coeffs,
        0.0
    )

    print(
        f"degree {degree}: "
        f"{zero_noise:.{precision}f}"
    )
