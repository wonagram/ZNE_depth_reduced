"""GPU2 / ZNE ver2 experiment plan (Rounds 1-4, increasing Pauli only).

IMPORTANT:
- This file REPLACES experiment_config_2.py only AFTER the old GPU2 batch
  and its worker processes have been stopped. Run with run_experiments_2.py.
- Jobs are DISPATCHED in Round 1 -> 2 -> 3 -> 4 order, then 2q -> 3q -> 4q
  -> 5q within a round. The scheduler can dispatch a later item while an
  earlier item is still running, so no GPU waits for an entire qubit/round
  barrier to finish.
- Only matching VALID existing reports are skipped. An existing but
  nonmatching/corrupted report stops the run rather than being overwritten.
- There is NO elapsed-time job timeout.
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
LOG_DIRECTORY = "./batch_logs_gpu2_ver2_multigpu"

METHOD_FILES = {
    "circuit_folding": "ZNE_circuit_folding.py",
    "global": "ZNE_depth_reduced_global.py",
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
        "phase_label": "GPU2 ZNE ver2 - increasing Pauli, Rounds 1-4",
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

# Separate per-qubit experiment groups allow the ORIGINAL base job builder
# to select exactly 3 methods for 2q/3q and 2 methods for 4q/5q.
EXPERIMENTS = [
    {
        "name": f"{unitary}_{qubits}q_ver2",
        "unitary": unitary,
        "methods": (
            ["circuit_folding", "global", "local"]
            if qubits in (2, 3)
            else ["circuit_folding", "global"]
        ),
        "depths_by_num_qubits": {qubits: depths},
    }
    for unitary, qubit_depths in DEPTHS_BY_UNITARY_AND_QUBITS.items()
    for qubits, depths in qubit_depths.items()
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
