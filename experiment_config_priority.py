"""Priority-ordered 16-stage, eight-GPU ZNE batch (Nighthawk r2 + Eagle r3).

ONE global priority queue: priority -> round -> qubit -> circuit -> method.
No priority/round/qubit completion barriers, no job timeouts, NO fake backend.
Heron r3 is preserved in the individual simulators but NOT in this queue.
"""

from __future__ import annotations

GPU_IDS = (0, 1, 2, 3, 4, 5, 6, 7)
SKIP_EXISTING = True  # ONLY when the exact target JSON exists in the task output folder.
LOG_DIRECTORY = "./batch_logs_priority"
HEARTBEAT_SECONDS = 60
STOP_ON_JOB_FAILURE = False  # record failures; keep other GPUs working.

METHOD_FILES = {
    "circuit_folding": "ZNE_circuit_folding.py",
    "global": "ZNE_depth_reduced_global.py",
    "local": "ZNE_depth_reduced_local.py",
}
METHOD_ORDER = ("circuit_folding", "global", "local")
CIRCUIT_ORDER = ("random_type", "hea_cz", "hea_crz", "hea_rzz")
QUBIT_ORDER = (2, 3, 4, 5)
ROUND_ORDER = (1, 2, 3, 4)
DEPTHS_BY_UNITARY_AND_QUBITS = {
    "random_type": {2: (24, 48, 72, 96), 3: (27, 54, 81, 108),
                    4: (32, 64, 96, 128), 5: (32, 64, 96, 128)},
    "hea_cz":      {2: (30, 60, 90, 120), 3: (32, 64, 96, 128),
                    4: (32, 64, 96, 128), 5: (32, 64, 96, 128)},
    "hea_crz":     {2: (30, 60, 90, 120), 3: (30, 60, 90, 120),
                    4: (30, 60, 90, 120), 5: (30, 60, 90, 120)},
    "hea_rzz":     {2: (30, 60, 90, 120), 3: (32, 64, 96, 128),
                    4: (32, 64, 96, 128), 5: (32, 64, 96, 128)},
}
FOLD_FACTORS_BY_QUBITS = {
    2: (1, 3, 5, 7),
    3: (1, 3, 5, 7),
    4: (1, 3, 5, 7),
    5: (1, 3, 5),  # no fold factor 7 to avoid 35 physical wires
}
SHOTS = 1000
BASE_SEED = 150
GATE_WEIGHT_1Q = 0.3
GATE_WEIGHT_2Q = 0.7
MAX_SIMULATOR_PHYSICAL_QUBITS = 40
CONSTANT_PAULI_DEPTH = 75
BACKEND = "aer_simulator"
MACHINE = "gpu"

# Folder naming rule:
# - Eagle: keep ALL existing directory names unchanged.
# - Nighthawk: add `_noise_model_nighthawk` after the corresponding
#   Eagle folder name, just as in the previously proposed Heron workflow.
# Report *filenames* include explicit suffixes _eagle / _nighthawk,
# optionally followed by _bell_noise.  No "ver" in report filenames/JSON.
# 'fast': all methods for 2q and 3q; only circuit/global for 4q and 5q.
# 'local45': deferred 4q/5q local ONLY, same output folder as fast.
# 'full': all methods for all qubit counts, Bell-ON benchmarks.

def stage(priority, label, noise, hardware, selection, *, bell_noise=False):
    version = {'increasing_random_pauli':'ver2',
               'increasing_exponential_random_pauli':'ver3',
               'constant_seeded_random_pauli':'ver4'}[noise]
    d = f"report_{version}" + ("_bell_noise" if bell_noise else "")
    if hardware == "nighthawk_r2":
        d += "_noise_model_nighthawk"
    assert hardware in ("eagle_r3", "nighthawk_r2")
    return dict(priority=priority, label=label, noise=noise, noise_model=hardware,
                selection=selection, bell_noise=bell_noise, directory=d)

L = "increasing_random_pauli"
X = "increasing_exponential_random_pauli"
C = "constant_seeded_random_pauli"
N = "nighthawk_r2"
E = "eagle_r3"

STAGES = (
    stage(1, "ver3_nighthawk", X, N, "fast"),
    stage(2, "ver4_nighthawk", C, N, "fast"),
    stage(3, "ver2_nighthawk", L, N, "fast"),
    stage(4, "ver3_eagle", X, E, "fast"),
    stage(5, "ver4_eagle", C, E, "fast"),
    stage(6, "ver3_nighthawk_local45", X, N, "local45"),
    stage(7, "ver4_nighthawk_local45", C, N, "local45"),
    stage(8, "ver2_nighthawk_local45", L, N, "local45"),
    stage(9, "ver3_eagle_local45", X, E, "local45"),
    stage(10, "ver4_eagle_local45", C, E, "local45"),
    stage(11, "ver2_nighthawk_bell_noise", L, N, "full", bell_noise=True),
    stage(12, "ver3_nighthawk_bell_noise", X, N, "full", bell_noise=True),
    stage(13, "ver4_nighthawk_bell_noise", C, N, "full", bell_noise=True),
    stage(14, "ver2_eagle_bell_noise", L, E, "full", bell_noise=True),
    stage(15, "ver3_eagle_bell_noise", X, E, "full", bell_noise=True),
    stage(16, "ver4_eagle_bell_noise", C, E, "full", bell_noise=True),
)

# Neither the configured queue nor its selection API includes Heron r3.
NOISE_MODEL_CHOICES = ("both", "eagle_r3", "nighthawk_r2")
INITIAL_PAULI_PROBABILITIES = {
    "eagle_r3": {"p_1q": 0.000245, "p_2q": 0.006970},
    "nighthawk_r2": {"p_1q": 0.0002114, "p_2q": 0.001720},
}


def selected_stages(noise_model_selection: str):
    if noise_model_selection not in NOISE_MODEL_CHOICES:
        raise ValueError(f"Unsupported batch profile {noise_model_selection!r}; "
                         f"must be one of {NOISE_MODEL_CHOICES}")
    if noise_model_selection == "both":
        return STAGES
    return tuple(s for s in STAGES if s["noise_model"] == noise_model_selection)


NOISE_FILENAME_TAGS = {
    L: "increasing_pauli",
    X: "increasing_exponential_pauli",
    C: f"constant_seeded_pauli_D{CONSTANT_PAULI_DEPTH}",
}


def selected_methods(selection: str, num_qubits: int) -> tuple[str, ...]:
    if selection == "fast":
        return METHOD_ORDER if num_qubits in (2, 3) else METHOD_ORDER[:2]
    if selection == "local45":
        return ("local",) if num_qubits in (4, 5) else ()
    if selection == "full":
        return METHOD_ORDER
    raise ValueError(f"Unknown stage selection: {selection}")
