"""GPU2 add-on: ONLY 4q/5q local width-folding, Rounds 1-4, increasing Pauli.

- 32 jobs total (4 circuits x 4 rounds x 2 qubit sizes), minus matching reports.
- This is a NEW companion configuration. Do NOT replace experiment_config_2.py.
- Reports are written into the existing zne_reports_ver2 directory.
- Existing reports are only skipped when matching; no existing JSON is overwritten.
- GPU scheduling has no round-completion barriers and no elapsed-time timeout.
"""

from __future__ import annotations

DRY_RUN = True
SKIP_EXISTING = True
STREAM_OUTPUT = False
PROGRESS_UPDATE_SECONDS = 30
STOP_ON_FAILURE = False
JOB_TIMEOUT_SECONDS = None  # ABSOLUTE RULE: never time out a long job.

GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
CUDA_VISIBLE_DEVICES = "0"  # compatibility with the base runner

# Keep the previously created reports directory (plural spelling).
# A singular name would not find already completed experiments.
REPORT_DIRECTORY = "./zne_reports_ver2"
LOG_DIRECTORY = "./batch_logs_gpu2_local45"

METHOD_FILES = {
    "local": "ZNE_depth_reduced_local.py",
}

CIRCUIT_ORDER = ["random_type", "hea_cz", "hea_crz", "hea_rzz"]

COMMON = {
    "machine": "gpu",
    "profiles": ["aer_increasing_pauli"],
    "shots": 1000,
    "fold_factors": [1, 3, 5, 7],
    "fold_factors_by_num_qubits": {
        5: [1, 3, 5],
    },
    "base_seed": 150,
    "gate_weight_1q": 0.3,
    "gate_weight_2q": 0.7,
    "max_simulator_physical_qubits": 40,
}

EXECUTION_PROFILES = {
    "aer_increasing_pauli": {
        "backend": "aer_simulator",
        "noise": "increasing_random_pauli",
        "report_tag": "GPU_increasing_pauli",
        "batch_phase": 0,
        "phase_label": "GPU2 additional 4q/5q local folding - increasing Pauli",
        "order_priority": 0,
    },
}

# Depths by ordinal round, faithfully retained from the original
# experiment_config.py plan (including the unitary-specific periods).
# Each list is [Round 1, Round 2, Round 3, Round 4].
DEPTHS_BY_UNITARY_AND_QUBITS = {
    "random_type": {
        2: [24, 48, 72, 96],
        3: [27, 54, 81, 108],
        4: [32, 64, 96, 128],
        5: [32, 64, 96, 128],
    },
    "hea_cz": {
        2: [30, 60, 90, 120],
        3: [32, 64, 96, 128],
        4: [32, 64, 96, 128],
        5: [32, 64, 96, 128],
    },
    "hea_crz": {
        2: [30, 60, 90, 120],
        3: [30, 60, 90, 120],
        4: [30, 60, 90, 120],
        5: [30, 60, 90, 120],
    },
    "hea_rzz": {
        2: [30, 60, 90, 120],
        3: [32, 64, 96, 128],
        4: [32, 64, 96, 128],
        5: [32, 64, 96, 128],
    },
}

# Generate ONLY the missing 4q and 5q local-folding experiment points.
# Preserve the original circuit-specific depth schedules and fold factors.
EXPERIMENTS = [
    {
        "name": f"{unitary}_{qubits}q_local45",
        "unitary": unitary,
        "methods": ["local"],
        "depths_by_num_qubits": {qubits: DEPTHS_BY_UNITARY_AND_QUBITS[unitary][qubits]},
    }
    for unitary in CIRCUIT_ORDER
    for qubits in (4, 5)
]

# Do not automatically stage, commit, or push logs or reports.
GIT = {
    "enabled": False,
    "push": False,
    "commit_every_successes": 10,
    "remote": "origin",
    "branch": None,
    "paths": [REPORT_DIRECTORY],
}
