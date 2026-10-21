"""Validation that does not require Qiskit or a GPU."""

from pathlib import Path
import importlib.util
import sys

import experiment_config as cfg
import run_experiments as runner
from zne_cli import resolve_backend_flags, resolve_machine_flags, resolve_noise_flags


ROOT = Path(__file__).resolve().parent


def load_u_config():
    spec = importlib.util.spec_from_file_location("validation_u_config", ROOT / "U_config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    uc = load_u_config()
    jobs = runner._build_jobs(
        ROOT,
        profile_filter=None,
        experiment_filter=None,
        method_filter=None,
    )
    warnings = runner._validate_jobs(jobs, uc)

    assert len(jobs) == 780, len(jobs)
    assert all(job.machine == "gpu" for job in jobs)
    assert len({job.report_path for job in jobs}) == len(jobs)

    # Requested HEA boundaries.
    expected = {
        "hea_cz": {2: [30, 60, 90, 120], 3: [32, 64, 96, 128],
                   4: [32, 64, 96, 128], 5: [32, 64, 96, 128]},
        "hea_rzz": {2: [30, 60, 90, 120], 3: [32, 64, 96, 128],
                    4: [32, 64, 96, 128], 5: [32, 64, 96, 128]},
        "hea_crz": {2: [36, 72, 108, 144], 3: [30, 60, 90, 120],
                    4: [30, 60, 90, 120], 5: [30, 60, 90, 120]},
    }
    for unitary, by_q in expected.items():
        for q, depths in by_q.items():
            for depth in depths:
                selected = uc.select_unitary_components(unitary, q, depth)
                assert selected[-1].included_depth == selected[-1].full_depth

    # random_type periods and maximums.
    assert uc.get_unitary_max_depth("random_type", 1) == 120
    assert uc.get_unitary_max_depth("random_type", 2) == 120
    assert uc.get_unitary_max_depth("random_type", 3) == 126
    assert uc.get_unitary_max_depth("random_type", 4) == 128

    assert resolve_machine_flags(
        "gpu", default_use_real_hardware=False, default_use_gpu=False
    ) == (False, True)
    assert resolve_backend_flags(
        "fake_backend", use_real_hardware=False,
        default_use_aer_simulator=True,
        default_use_statevector=False,
        default_use_fake_backend=False,
    ) == (False, False, True)
    assert resolve_noise_flags(
        "increasing_random_pauli_amplitude_damping",
        defaults=(True, False, False, False, False, False),
    ) == (False, False, False, False, True, True)

    print(f"PASS: {len(jobs)} unique jobs validated.")
    print("PASS: all requested HEA depths end on complete components.")
    print("PASS: random_type maximum depths and CLI mappings validated.")
    for warning in dict.fromkeys(warnings):
        print("EXPECTED WARNING:", warning)


if __name__ == "__main__":
    main()
