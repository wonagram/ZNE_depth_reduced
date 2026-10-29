"""GPU-2 batch configuration: run only original Round 3 and Round 4.

This file intentionally keeps the same experiment semantics as the primary
experiment_config.py, but isolates output paths and restricts the depth sweep
to the third and fourth depth rounds only.

Execution order inside each phase is strictly:
    Round 3: 1q -> 2q -> 3q -> 4q -> 5q
    Round 4: 1q -> 2q -> 3q -> 4q -> 5q
and, within each qubit count:
    random_type -> hea_cz -> hea_crz -> hea_rzz

There is NO per-job timeout.
"""

from __future__ import annotations

# ============================================================
# Safety and execution behavior
# ============================================================
DRY_RUN = True
SKIP_EXISTING = True
STREAM_OUTPUT = False
PROGRESS_UPDATE_SECONDS = 30
STOP_ON_FAILURE = False

# Never terminate a long-running job because of elapsed time.
JOB_TIMEOUT_SECONDS = None

# On a separate GPU machine with one visible GPU, keep "0".
# If this config is used on the same multi-GPU host and GPU 0 is occupied,
# change this to the desired visible device index, e.g. "1".
CUDA_VISIBLE_DEVICES = "0"

# ============================================================
# Separate GPU-2 output paths: do not overlap GPU-1 output directories
# ============================================================
REPORT_DIRECTORY = "./zne_reports_gpu2_round3_round4"
LOG_DIRECTORY = "./batch_logs_gpu2_round3_round4"

METHOD_FILES = {
    "circuit_folding": "ZNE_circuit_folding.py",
    "global": "ZNE_depth_reduced_global.py",
    "local": "ZNE_depth_reduced_local.py",
}

# ============================================================
# Settings shared by all GPU-2 experiments
# ============================================================
COMMON = {
    "machine": "gpu",
    "methods": "all",
    "profiles": [
        "aer_small_constant_depolarizing_0p2pct",
        "aer_increasing_pauli",
        "aer_increasing_pauli_amplitude_damping",
        "fake_quebec_statevector",
    ],
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

# ============================================================
# Backend/noise profiles
# ============================================================
EXECUTION_PROFILES = {
    # Phase 0: custom Aer jobs, Round 3 then Round 4.
    "aer_small_constant_depolarizing_0p2pct": {
        "backend": "aer_simulator",
        "noise": "constant_depolarizing",
        "depolarizing_strength": 0.002,  # 0.2%
        "report_tag": "GPU_small_constant_depolarizing_0p2pct",
        "batch_phase": 0,
        "phase_label": "GPU2 PHASE A - Aer Round 3 and Round 4",
        "order_priority": 0,
    },
    "aer_increasing_pauli": {
        "backend": "aer_simulator",
        "noise": "increasing_random_pauli",
        "report_tag": "GPU_increasing_pauli",
        "batch_phase": 0,
        "phase_label": "GPU2 PHASE A - Aer Round 3 and Round 4",
        "order_priority": 1,
    },
    "aer_increasing_pauli_amplitude_damping": {
        "backend": "aer_simulator",
        "noise": "increasing_random_pauli_amplitude_damping",
        "report_tag": "GPU_increasing_pauli_amplitude_damping",
        "batch_phase": 0,
        "phase_label": "GPU2 PHASE A - Aer Round 3 and Round 4",
        "order_priority": 2,
    },

    # Phase 1: full FakeQuebec jobs for the same Round 3/4 depth points.
    "fake_quebec_statevector": {
        "backend": "fake_statevector",
        "fake_backend_name": "FakeQuebec",
        "noise": "none",
        "report_tag": "GPU_FakeQuebec_statevector",
        "allowed_num_qubits": [2, 3, 4, 5],
        "batch_phase": 1,
        "phase_label": "GPU2 PHASE B - FakeQuebec Round 3 and Round 4",
        "order_priority": 0,
    },
}

# ============================================================
# ONLY ORIGINAL ROUND 3 AND ROUND 4
#
# depth_round_offset=2 is consumed by run_experiments_2.py so the two retained
# depth positions are displayed and sorted as Round 3 and Round 4, not as a new
# Round 1 and Round 2.
# ============================================================
EXPERIMENTS = [
    {
        "name": "random_type_round3_round4",
        "unitary": "random_type",
        "depth_round_offset": 2,
        "depths_by_num_qubits": {
            1: [90, 120],
            2: [72, 96],
            3: [81, 108],
            4: [96, 128],
            5: [96, 128],
        },
    },
    {
        "name": "hea_cz_round3_round4",
        "unitary": "hea_cz",
        "depth_round_offset": 2,
        "depths_by_num_qubits": {
            2: [90, 120],
            3: [96, 128],
            4: [96, 128],
            5: [96, 128],
        },
    },
    {
        "name": "hea_crz_round3_round4",
        "unitary": "hea_crz",
        "depth_round_offset": 2,
        "depths_by_num_qubits": {
            2: [90, 120],
            3: [90, 120],
            4: [90, 120],
            5: [90, 120],
        },
    },
    {
        "name": "hea_rzz_round3_round4",
        "unitary": "hea_rzz",
        "depth_round_offset": 2,
        "depths_by_num_qubits": {
            2: [90, 120],
            3: [96, 128],
            4: [96, 128],
            5: [96, 128],
        },
    },
]

# ============================================================
# Git checkpointing
# ============================================================
GIT = {
    "enabled": False,
    "push": False,
    "commit_every_successes": 10,
    "remote": "origin",
    "branch": None,
    "paths": [REPORT_DIRECTORY, LOG_DIRECTORY],
}
