from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit.circuit import Gate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import matplotlib.pyplot as plt

import json
from pathlib import Path
from scipy.optimize import curve_fit

from U_config import (
    get_unitary_max_depth,
    normalize_unitary_type,
    select_unitary_components,
    selected_unitary_summary,
)
from unitary_tools import (
    NoiseController,
    append_unitary_block,
    configure_aer_simulator_device,
    make_fake_aer_backend,
)
from state_preparation import *

# ============================================================
# Execution machine
# ============================================================
# Real hardware and local simulation are mutually exclusive.
# For Aer simulation, USE_GPU=True selects Aer device="GPU".
# When USE_GPU=False, local simulation uses the CPU.
USE_REAL_HARDWARE = False
USE_GPU = False

REAL_HARDWARE_DEVICE_NAME = False
# REAL_HARDWARE_DEVICE_NAME = "ibm_strasbourg"

# ============================================================
# Simulation backend
# ============================================================
# These flags apply only when USE_REAL_HARDWARE=False.
# Select exactly one of the three simulation backends.
USE_AER_SIMULATOR = True       # shot-based Aer, with optional custom noise
USE_STATEVECTOR = False        # exact noiseless Aer statevector
USE_FAKE_BACKEND = False       # IBM fake-device snapshot executed by Aer

# Fake-backend mode uses an IBM fake-device snapshot to configure Aer.
# The snapshot supplies the device noise model; custom channels are disabled.
FAKE_BACKEND_NAME = "FakeQuebec"
# Available modern examples (qiskit_ibm_runtime.fake_provider):
#   FakeSherbrooke, FakeBrisbane, FakeKyiv, FakeKyoto, FakeOsaka,
#   FakeQuebec, FakeStrasbourg, FakeTorino, FakeBrussels, FakeKawasaki,
#   FakeCusco, FakeFez, FakeMarrakesh, FakeAachen, FakeBerlin,
#   FakeBoston, FakeKingston, FakeMiami, FakePittsburgh
#   FakeFractionalBackend explicitly models control flow, but has only 5 qubits.

_SIMULATION_BACKEND_FLAGS = {
    "USE_AER_SIMULATOR": USE_AER_SIMULATOR,
    "USE_STATEVECTOR": USE_STATEVECTOR,
    "USE_FAKE_BACKEND": USE_FAKE_BACKEND,
}
_ENABLED_SIMULATION_BACKENDS = [
    name for name, enabled in _SIMULATION_BACKEND_FLAGS.items() if enabled
]

if USE_REAL_HARDWARE:
    if USE_GPU:
        raise ValueError(
            "USE_GPU cannot be combined with USE_REAL_HARDWARE. GPU/CPU selects "
            "the machine for Aer simulation, not the IBM quantum device."
        )
    if _ENABLED_SIMULATION_BACKENDS:
        raise ValueError(
            "When USE_REAL_HARDWARE=True, turn off USE_AER_SIMULATOR, "
            "USE_STATEVECTOR, and USE_FAKE_BACKEND. Enabled simulation backends: "
            + ", ".join(_ENABLED_SIMULATION_BACKENDS)
        )
    SIMULATION_DEVICE = None
else:
    if len(_ENABLED_SIMULATION_BACKENDS) != 1:
        raise ValueError(
            "When USE_REAL_HARDWARE=False, select exactly one simulation backend: "
            "USE_AER_SIMULATOR, USE_STATEVECTOR, or USE_FAKE_BACKEND. Enabled: "
            + (", ".join(_ENABLED_SIMULATION_BACKENDS)
               if _ENABLED_SIMULATION_BACKENDS else "none")
        )
    SIMULATION_DEVICE = "GPU" if USE_GPU else "CPU"

shots = 10**3  # Used by shot-based Aer/fake/hardware; ignored by statevector.

# ============================================================
# Gate-count-axis extrapolation report
# ============================================================
# Four extrapolation axes are always reported:
#   1) before transpilation: N = w1*N_1q + w2*N_2q
#   2) before transpilation: N = N_2q
#   3) after transpilation:  N = w1*N_1q + w2*N_2q
#   4) after transpilation:  N = N_2q
#
# The report is JSON and is overwritten when the same qubit/unitary/method
# combination is run again.
REPORT_DIRECTORY = "./zne_reports"
GATE_WEIGHT_1Q = 0.3
GATE_WEIGHT_2Q = 0.7


# ============================================================
# Unitary selection, counted depth, and logical-qubit count
# Available UNITARY_TYPE values:
#   random_type, hea_cz, hea_crx, hea_crz, hea_rzz,
#   qaoa, qaoa_xy, clifford_only
# ============================================================
UNITARY_TYPE = "random_type"
DEPTH = 18
NUM_U_QUBITS = 2

# ============================================================
# Circuit drawing
# ============================================================
DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 3

# ============================================================
# Number of foldings
# ============================================================
depth_folded_circuits = [1, 3, 5, 7]

# ============================================================
# Custom noise configuration for USE_AER_SIMULATOR
#
# Historical constant-noise mode (preserved):
#   select any combination of the four USE_CONSTANT_* flags.
#
# New depth-dependent mode:
#   select USE_INCREASING_RANDOM_PAULI and/or
#   USE_INCREASING_AMPLITUDE_DAMPING.
#
# The historical and depth-dependent families are mutually exclusive.
# Statevector, fake-backend, and real-hardware modes cannot be combined with
# custom-noise flags.  Aer with every custom-noise flag False is noiseless,
# shot-based Aer and remains available for backward compatibility.
# ============================================================
USE_CONSTANT_DEPOLARIZING = True
USE_CONSTANT_AMPLITUDE_DAMPING = False
USE_CONSTANT_PHASE_DAMPING = False
USE_CONSTANT_COHERENT_OVERROTATION = False

USE_INCREASING_RANDOM_PAULI = False
USE_INCREASING_AMPLITUDE_DAMPING = False

# Historical constant-noise strengths (same semantics as before).
depolarizing_strength = 0.02
amplitude_damping_strength = 0.02
phase_damping_strength = 0.02
overrotation_epsilon = 0.02

# Reproducible depth-dependent random-Pauli assignments and Aer sampling.
BASE_SEED = 150

# p(D) = p0 * [1 + (NOISE_MULTIPLIER - 1) * D / REFERENCE_DEPTH].
# The model is not capped beyond REFERENCE_DEPTH.  An explicit error is raised
# if any probability becomes larger than 1.
REFERENCE_DEPTH = 50
NOISE_MULTIPLIER = 10.0

P1_INITIAL = 0.000245
P2_INITIAL = 0.006970
GAMMA1_INITIAL = 0.000112173
GAMMA2_INITIAL = 0.002113149

# Displayed slopes implied by the formula above.  Change the initial values,
# REFERENCE_DEPTH, or NOISE_MULTIPLIER to change these slopes consistently.
P1_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * P1_INITIAL / REFERENCE_DEPTH
)
P2_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * P2_INITIAL / REFERENCE_DEPTH
)
GAMMA1_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * GAMMA1_INITIAL / REFERENCE_DEPTH
)
GAMMA2_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * GAMMA2_INITIAL / REFERENCE_DEPTH
)

_CONSTANT_NOISE_ACTIVE = any((
    USE_CONSTANT_DEPOLARIZING,
    USE_CONSTANT_AMPLITUDE_DAMPING,
    USE_CONSTANT_PHASE_DAMPING,
    USE_CONSTANT_COHERENT_OVERROTATION,
))
_INCREASING_NOISE_ACTIVE = any((
    USE_INCREASING_RANDOM_PAULI,
    USE_INCREASING_AMPLITUDE_DAMPING,
))
_CUSTOM_NOISE_ACTIVE = _CONSTANT_NOISE_ACTIVE or _INCREASING_NOISE_ACTIVE

if _CONSTANT_NOISE_ACTIVE and _INCREASING_NOISE_ACTIVE:
    raise ValueError(
        "Do not enable historical constant noise and depth-dependent noise "
        "simultaneously. Choose one custom-noise family."
    )

if _CUSTOM_NOISE_ACTIVE and not USE_AER_SIMULATOR:
    raise ValueError(
        "Custom constant/depth-dependent noise is available only with "
        "USE_AER_SIMULATOR=True. Turn off custom-noise flags when using "
        "statevector, a fake backend, or real hardware."
    )

# ============================================================
# Select U. DEPTH may cut any decomposed macro at a physical-depth boundary.
# ============================================================
UNITARY_TYPE = normalize_unitary_type(UNITARY_TYPE)
SELECTED_COMPONENTS = select_unitary_components(
    UNITARY_TYPE,
    NUM_U_QUBITS,
    DEPTH,
)
MAX_U_DEPTH = get_unitary_max_depth(UNITARY_TYPE, NUM_U_QUBITS)

NOISE_CONTROLLER = NoiseController(
    # Historical constant mode.
    use_depolarizing=USE_CONSTANT_DEPOLARIZING,
    use_amplitude_damping=USE_CONSTANT_AMPLITUDE_DAMPING,
    use_phase_damping=USE_CONSTANT_PHASE_DAMPING,
    use_coherent_overrotation=USE_CONSTANT_COHERENT_OVERROTATION,
    depolarizing_strength=depolarizing_strength,
    amplitude_damping_strength=amplitude_damping_strength,
    phase_damping_strength=phase_damping_strength,
    overrotation_epsilon=overrotation_epsilon,
    # New depth-dependent mode.
    use_increasing_random_pauli=USE_INCREASING_RANDOM_PAULI,
    use_increasing_amplitude_damping=USE_INCREASING_AMPLITUDE_DAMPING,
    base_seed=BASE_SEED,
    reference_depth=REFERENCE_DEPTH,
    noise_multiplier=NOISE_MULTIPLIER,
    p1_initial=P1_INITIAL,
    p2_initial=P2_INITIAL,
    gamma1_initial=GAMMA1_INITIAL,
    gamma2_initial=GAMMA2_INITIAL,
)


def get_block(q, block_index):
    start = block_index * NUM_U_QUBITS
    end = start + NUM_U_QUBITS
    return [q[i] for i in range(start, end)]


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
# Gate counting, fitting, and JSON reporting
# ============================================================
_SKIP_COUNT_NAMES = {
    "barrier", "delay", "measure", "reset",
    "kraus", "superop", "quantum_channel", "pauli",
    "save_expectation_value", "save_statevector", "set_statevector",
}


def _count_one_two_qubit_gates(circuit, _level=0):
    """Count structural one- and two-qubit gates in a circuit.

    Control-flow blocks are inspected recursively.  Noise-channel instructions,
    measurement/reset, barriers, delays, and Aer save/set instructions are not
    counted as gates.  A composite operation acting on more than two qubits is
    recursively expanded when it exposes a definition.
    """
    if _level > 30:
        raise RuntimeError("Gate-count recursion exceeded 30 levels.")

    n1q = 0
    n2q = 0

    for item in circuit.data:
        op = item.operation
        name = getattr(op, "name", "")

        if getattr(op, "label", None) == "__zne_artificial_noise__":
            continue

        blocks = tuple(getattr(op, "blocks", ()) or ())
        if blocks:
            for block in blocks:
                c1, c2 = _count_one_two_qubit_gates(block, _level + 1)
                n1q += c1
                n2q += c2
            continue

        if name in _SKIP_COUNT_NAMES:
            continue

        if isinstance(op, Gate):
            width = len(item.qubits)
            if width == 1:
                n1q += 1
            elif width == 2:
                n2q += 1
            else:
                definition = getattr(op, "definition", None)
                if definition is not None:
                    c1, c2 = _count_one_two_qubit_gates(definition, _level + 1)
                    n1q += c1
                    n2q += c2
            continue

        definition = getattr(op, "definition", None)
        if definition is not None:
            c1, c2 = _count_one_two_qubit_gates(definition, _level + 1)
            n1q += c1
            n2q += c2

    return int(n1q), int(n2q)


def _gate_count_list(circuits):
    return [
        {"N_1q": n1q, "N_2q": n2q}
        for n1q, n2q in (_count_one_two_qubit_gates(circuit) for circuit in circuits)
    ]


def _linear_fit(N_values, expectations):
    x = np.asarray(N_values, dtype=float)
    y = np.asarray(expectations, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return {
            "success": False,
            "error": "Linear fitting requires at least two distinct N values.",
        }

    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    return {
        "success": True,
        "model": "a*N + b",
        "slope": float(slope),
        "intercept": float(intercept),
        "at_N_zero": float(intercept),
        "rmse": rmse,
    }


def _exponential_model(N, amplitude, decay_rate):
    return amplitude * np.exp(-decay_rate * N)


def _exponential_fit(N_values, expectations):
    x = np.asarray(N_values, dtype=float)
    y = np.asarray(expectations, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return {
            "success": False,
            "error": "Exponential fitting requires at least two distinct N values.",
        }

    smallest_index = int(np.argmin(x))
    amplitude0 = float(y[smallest_index])
    if abs(amplitude0) < 1e-12:
        amplitude0 = float(np.mean(y))
    if abs(amplitude0) < 1e-12:
        amplitude0 = 1e-6

    # Keep exp(-g N) numerically bounded over the observed gate-count range.
    scale = max(float(np.max(np.abs(x))), 1.0)
    g_limit = 40.0 / scale

    try:
        params, _ = curve_fit(
            _exponential_model,
            x,
            y,
            p0=[amplitude0, 0.0],
            bounds=([-np.inf, -g_limit], [np.inf, g_limit]),
            maxfev=50000,
        )
        amplitude, decay_rate = (float(params[0]), float(params[1]))
        fitted = _exponential_model(x, amplitude, decay_rate)
        rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
        return {
            "success": True,
            "model": "A*exp(-g*N)",
            "amplitude": amplitude,
            "decay_rate": decay_rate,
            "at_N_zero": amplitude,
            "rmse": rmse,
        }
    except Exception as error:
        return {
            "success": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _build_axis_case(case_name, N_definition, N_values, expectations):
    N_values = [float(value) for value in N_values]
    return {
        "case": case_name,
        "N_definition": N_definition,
        "N": N_values,
        "linear": _linear_fit(N_values, expectations),
        "exponential": _exponential_fit(N_values, expectations),
    }


def _build_gate_axis_report(raw_counts, transpiled_counts, expectations):
    raw_1q = [row["N_1q"] for row in raw_counts]
    raw_2q = [row["N_2q"] for row in raw_counts]
    transpiled_1q = [row["N_1q"] for row in transpiled_counts]
    transpiled_2q = [row["N_2q"] for row in transpiled_counts]

    raw_weighted = [
        GATE_WEIGHT_1Q * n1 + GATE_WEIGHT_2Q * n2
        for n1, n2 in zip(raw_1q, raw_2q)
    ]
    transpiled_weighted = [
        GATE_WEIGHT_1Q * n1 + GATE_WEIGHT_2Q * n2
        for n1, n2 in zip(transpiled_1q, transpiled_2q)
    ]

    weighted_definition = (
        f"N = {GATE_WEIGHT_1Q}*N_1q + {GATE_WEIGHT_2Q}*N_2q"
    )

    cases = {
        "before_weighted": _build_axis_case(
            "before_transpilation_weighted",
            weighted_definition,
            raw_weighted,
            expectations,
        ),
        "before_two_qubit": _build_axis_case(
            "before_transpilation_two_qubit",
            "N = N_2q",
            raw_2q,
            expectations,
        ),
        "after_weighted": _build_axis_case(
            "after_transpilation_weighted",
            weighted_definition,
            transpiled_weighted,
            expectations,
        ),
        "after_two_qubit": _build_axis_case(
            "after_transpilation_two_qubit",
            "N = N_2q",
            transpiled_2q,
            expectations,
        ),
    }

    return {
        "gate_counts": {
            "before_transpilation": {
                "N_1q": raw_1q,
                "N_2q": raw_2q,
                "N_weighted": [float(v) for v in raw_weighted],
            },
            "after_transpilation": {
                "N_1q": transpiled_1q,
                "N_2q": transpiled_2q,
                "N_weighted": [float(v) for v in transpiled_weighted],
            },
        },
        "cases": cases,
    }


def _print_gate_axis_results(axis_report, precision):
    print("\nGate-count-axis extrapolation to N = 0")
    order = [
        "before_weighted",
        "before_two_qubit",
        "after_weighted",
        "after_two_qubit",
    ]
    for key in order:
        case = axis_report["cases"][key]
        print(f"\n{key}: {case['N_definition']}")
        print("N = [" + ", ".join(f"{value:.6g}" for value in case["N"]) + "]")
        for fit_name in ("linear", "exponential"):
            fit = case[fit_name]
            if fit.get("success"):
                print(
                    f"{fit_name}: {fit['at_N_zero']:.{precision}f} at N=0 "
                    f"(RMSE={fit['rmse']:.3e})"
                )
            else:
                print(f"{fit_name}: FIT FAILED - {fit.get('error', 'unknown error')}")


def _safe_filename_tag(value):
    text = str(value).strip()
    for old, new in (
        (" ", "_"),
        ("(", ""),
        (")", ""),
        (":", "_"),
        ("/", "_"),
        ("\\", "_"),
    ):
        text = text.replace(old, new)
    return text or "unknown"


def _real_hardware_backend_name():
    if not USE_REAL_HARDWARE:
        return None
    name = backend.name
    if callable(name):
        name = name()
    return str(name)


def _backend_mode_name():
    if USE_REAL_HARDWARE:
        return f"real_hardware:{_real_hardware_backend_name()}"
    if USE_STATEVECTOR:
        return "statevector_noiseless"
    if USE_FAKE_BACKEND:
        return f"fake_backend:{FAKE_BACKEND_NAME}"
    if _INCREASING_NOISE_ACTIVE:
        return "aer_depth_dependent_noise"
    if _CONSTANT_NOISE_ACTIVE:
        return "aer_artificial_noise"
    return "aer_shot_noiseless"


def _backend_kind_for_report():
    if USE_REAL_HARDWARE:
        return "real_hardware"
    if USE_STATEVECTOR:
        return "statevector"
    if USE_FAKE_BACKEND:
        return "fake_backend"
    return "aer_simulator"


def _backend_name_for_report():
    if USE_REAL_HARDWARE:
        return _real_hardware_backend_name()
    if USE_STATEVECTOR:
        return "AerSimulator(statevector)"
    if USE_FAKE_BACKEND:
        return str(FAKE_BACKEND_NAME)
    return "AerSimulator"


def _machine_name_for_report():
    return "real_hardware" if USE_REAL_HARDWARE else str(SIMULATION_DEVICE)


def _execution_filename_tag():
    """Filename tag for machine/backend without a redundant AerSimulator label.

    Examples:
      plain Aer CPU/GPU -> CPU or GPU
      statevector       -> CPU_statevector or GPU_statevector
      fake backend      -> CPU_FakeQuebec or GPU_FakeQuebec
      real hardware     -> ibm_backend_name
    """
    if USE_REAL_HARDWARE:
        return _safe_filename_tag(_real_hardware_backend_name())

    parts = [str(SIMULATION_DEVICE)]
    if USE_STATEVECTOR:
        parts.append("statevector")
    elif USE_FAKE_BACKEND:
        parts.append(_safe_filename_tag(FAKE_BACKEND_NAME))
    return "_".join(parts)


def _noise_report():
    active_constant_types = []
    if USE_AER_SIMULATOR and USE_CONSTANT_DEPOLARIZING:
        active_constant_types.append("constant_depolarizing")
    if USE_AER_SIMULATOR and USE_CONSTANT_AMPLITUDE_DAMPING:
        active_constant_types.append("constant_amplitude_damping")
    if USE_AER_SIMULATOR and USE_CONSTANT_PHASE_DAMPING:
        active_constant_types.append("constant_phase_damping")
    if USE_AER_SIMULATOR and USE_CONSTANT_COHERENT_OVERROTATION:
        active_constant_types.append("constant_coherent_overrotation")

    active_increasing_types = []
    if USE_AER_SIMULATOR and USE_INCREASING_RANDOM_PAULI:
        active_increasing_types.append("depth_dependent_random_pauli")
    if USE_AER_SIMULATOR and USE_INCREASING_AMPLITUDE_DAMPING:
        active_increasing_types.append("depth_dependent_amplitude_damping")

    active_custom_types = active_constant_types + active_increasing_types

    if USE_FAKE_BACKEND:
        source = "fake_backend_device_snapshot"
    elif USE_REAL_HARDWARE:
        source = "real_hardware_device_noise"
    elif USE_STATEVECTOR:
        source = "none_statevector_noiseless"
    elif active_increasing_types:
        source = "depth_dependent_artificial_noise_channels"
    elif active_constant_types:
        source = "constant_artificial_noise_channels"
    else:
        source = "none_aer_shot_noiseless"

    return {
        "source": source,
        "selected_custom_noise_family": (
            "depth_dependent" if active_increasing_types
            else "constant" if active_constant_types
            else None
        ),
        "custom_noise_channels_applied": bool(active_custom_types),
        # Kept for compatibility with earlier reports.
        "active_custom_noise_types": active_custom_types,
        "active_constant_noise_types": active_constant_types,
        "active_depth_dependent_noise_types": active_increasing_types,
        "constant_noise_parameters": {
            "depolarizing_strength": float(depolarizing_strength),
            "amplitude_damping_strength": float(amplitude_damping_strength),
            "phase_damping_strength": float(phase_damping_strength),
            "coherent_overrotation_epsilon": float(overrotation_epsilon),
        },
        # Kept for compatibility with earlier reports.
        "custom_noise_parameters": {
            "depolarizing_strength": float(depolarizing_strength),
            "amplitude_damping_strength": float(amplitude_damping_strength),
            "phase_damping_strength": float(phase_damping_strength),
            "coherent_overrotation_epsilon": float(overrotation_epsilon),
        },
        "depth_dependent_noise_configuration": (
            NOISE_CONTROLLER.runtime_report() if _INCREASING_NOISE_ACTIVE else None
        ),
        "fake_backend_name": str(FAKE_BACKEND_NAME) if USE_FAKE_BACKEND else None,
    }


def _write_report(method_name, expectations, raw_counts, transpiled_counts,
                  ideal_expectation=None):
    axis_report = _build_gate_axis_report(
        raw_counts,
        transpiled_counts,
        expectations,
    )

    report = {
        "schema_version": 4,
        "method": method_name,
        "num_qubits": int(NUM_U_QUBITS),
        "unitary_type": str(UNITARY_TYPE),
        "depth": int(DEPTH),
        "fold_factors": [int(value) for value in depth_folded_circuits],
        "expectation_values": [float(value) for value in expectations],
        "ideal_expectation": (
            None if ideal_expectation is None else float(ideal_expectation)
        ),
        "backend_mode": _backend_mode_name(),
        "execution": {
            "machine": _machine_name_for_report(),
            "simulation_device": (
                None if USE_REAL_HARDWARE else str(SIMULATION_DEVICE)
            ),
            "gpu_enabled": bool(USE_GPU and not USE_REAL_HARDWARE),
            "backend_kind": _backend_kind_for_report(),
            "backend_mode": _backend_mode_name(),
            "backend_name": _backend_name_for_report(),
            "filename_tag": _execution_filename_tag(),
            "shots": None if USE_STATEVECTOR else int(shots),
        },
        "noise": _noise_report(),
        "weights": {
            "w1_one_qubit": float(GATE_WEIGHT_1Q),
            "w2_two_qubit": float(GATE_WEIGHT_2Q),
        },
        "gate_count_definition": (
            "Structural gate count. Measurements, resets, barriers, delays, "
            "Aer save/set instructions, and explicit artificial-noise-channel "
            "instructions are excluded. Control-flow blocks are counted recursively."
        ),
        **axis_report,
    }

    output_directory = Path(REPORT_DIRECTORY).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    execution_tag = _execution_filename_tag()

    filename = (
        f"{NUM_U_QUBITS}q_{UNITARY_TYPE}_{DEPTH}_"
        f"{execution_tag}_{method_name}.json"
    )
    path = output_directory / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)

    return path, axis_report

# ============================================================
# Backend
# ============================================================


if USE_REAL_HARDWARE:

    if not QiskitRuntimeService.saved_accounts():
        raise RuntimeError(
            "USE_REAL_HARDWARE=True, but no IBM Quantum account is saved."
        )

    service = QiskitRuntimeService(channel="ibm_quantum")

    if REAL_HARDWARE_DEVICE_NAME:
        backend = service.backend(REAL_HARDWARE_DEVICE_NAME)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            dynamic_circuits=True,
        )

elif USE_STATEVECTOR:
    backend = AerSimulator(
        method="statevector",
        precision="double",
        zero_threshold=0.0,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    configure_aer_simulator_device(backend, use_gpu=USE_GPU)

elif USE_FAKE_BACKEND:

    FAKE_DEVICE, backend = make_fake_aer_backend(
        FAKE_BACKEND_NAME,
        seed_simulator=BASE_SEED,
        required_qubits=NUM_U_QUBITS * max(depth_folded_circuits),
        use_gpu=USE_GPU,
    )

elif USE_AER_SIMULATOR:

    backend = AerSimulator()
    configure_aer_simulator_device(backend, use_gpu=USE_GPU)
    backend.set_options(seed_simulator=BASE_SEED)

else:  # Protected by the machine/backend validation above.
    raise AssertionError("No execution backend was selected.")

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

        odd_block = get_block(q, 2 * fold + 1)
        even_block = get_block(q, 2 * fold + 2)
        for k in range(NUM_U_QUBITS):
            psi_minus(
                qc,
                [odd_block[k], even_block[k]]
            )

    qc.barrier()

    # ========================================================
    # 3. Apply U to every qubit
    # ========================================================
    for block_index in range(depth_folded):
        block = get_block(q,block_index)
        compile_stats = append_unitary_block(
            qc,
            block,
            SELECTED_COMPONENTS,
            inverse=False,
            dress_cx=(block_index % 2 == 1),
            noise_controller=NOISE_CONTROLLER,
            use_noise=(
                USE_AER_SIMULATOR
                and _CUSTOM_NOISE_ACTIVE
                and not DRAW_CIRCUIT
                and not USE_STATEVECTOR
            ),
            compile_cancellations=True,
        )
        for key, value in compile_stats.items():
            COMPILATION_TOTALS[key] += value
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

        even_block = get_block(q, 2 * fold)
        odd_block = get_block(q, 2 * fold + 1)

        for k in range(NUM_U_QUBITS):
            bell_measure(
                qc,
                [even_block[k], odd_block[k]]
            )

    qc.barrier()


    # ========================================================
    # 5. All Bell measurements
    # Do not append steps 4 and 5.
    # If we append them, the measurements will be performed sequentially.
    # ========================================================
    for fold in range(num_fold):

        even_block = get_block(q, 2 * fold)
        odd_block = get_block(q, 2 * fold + 1)

        for k in range(NUM_U_QUBITS):
            qA = even_block[k]
            qB = odd_block[k]

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
                even_block = get_block(q, 2 * fold)
                odd_block = get_block(q, 2 * fold + 1)
                apply_coherent_correction(
                    qc,
                    even_block[k],  # qA: originally c_pair[1]
                    odd_block[k],  # qB: originally c_pair[0]
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

print(f"\nUNITARY_TYPE = {UNITARY_TYPE}")
print(selected_unitary_summary(UNITARY_TYPE, NUM_U_QUBITS, DEPTH))
print(f"DEPTH = {DEPTH}; maximum configured depth = {MAX_U_DEPTH}")
print(
    "One-pass U compilation removed pairs: "
    f"Y={COMPILATION_TOTALS['global_y_pairs']}, "
    f"H={COMPILATION_TOTALS['h_pairs']}, "
    f"inverse RX(pi/2)={COMPILATION_TOTALS['basis_rx_pairs']}"
)

if USE_REAL_HARDWARE:
    print("Machine: real quantum hardware")
    print(f"Backend: {_real_hardware_backend_name()}")
else:
    print(f"Machine: {SIMULATION_DEVICE}")

    if USE_STATEVECTOR:
        print("Backend: AerSimulator (statevector, noiseless)")
    elif USE_FAKE_BACKEND:
        print(f"Backend: {FAKE_BACKEND_NAME} snapshot via AerSimulator")
        print("Noise: fake-backend device snapshot; custom noise channels disabled")
    elif USE_AER_SIMULATOR:
        print("Backend: AerSimulator")

        active_noises = []
        if USE_CONSTANT_DEPOLARIZING:
            active_noises.append("constant depolarizing")
        if USE_CONSTANT_AMPLITUDE_DAMPING:
            active_noises.append("constant amplitude damping")
        if USE_CONSTANT_PHASE_DAMPING:
            active_noises.append("constant phase damping")
        if USE_CONSTANT_COHERENT_OVERROTATION:
            active_noises.append("constant coherent overrotation")
        if USE_INCREASING_RANDOM_PAULI:
            active_noises.append("depth-dependent random Pauli")
        if USE_INCREASING_AMPLITUDE_DAMPING:
            active_noises.append("depth-dependent amplitude damping")

        print("Noise:", ", ".join(active_noises) if active_noises else "none")
    else:
        raise AssertionError("Unexpected simulator-backend selection while printing configuration.")

precision = 12 if USE_STATEVECTOR else 4
print("\nExpectation values of global method:")
print("[" + ", ".join(f"{x:.{precision}f}" for x in expectation_values) + "]")


# ============================================================
# Gate counts before/after transpilation
# ============================================================
raw_gate_counts = _gate_count_list(folded_circuits)
transpiled_gate_counts = _gate_count_list(exec_circuits)

# ============================================================
# Linear and exponential extrapolation on four gate-count axes
# ============================================================
report_path, axis_report = _write_report(
    "global",
    expectation_values,
    raw_gate_counts,
    transpiled_gate_counts,
    ideal_expectation=None,
)
_print_gate_axis_results(axis_report, precision)
print(f"\nReport saved to: {report_path}")
