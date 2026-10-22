"""Validate the capped-noise, per-qubit-fold ZNE batch configuration."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import run_experiments as runner
from unitary_tools import depth_noise_parameters, depth_noise_scale


def main() -> int:
    project_root = Path(__file__).resolve().parent
    u_config = runner._load_u_config(project_root)
    jobs = runner._build_jobs(
        project_root,
        profile_filter=None,
        experiment_filter=None,
        method_filter=None,
    )
    warnings = runner._validate_jobs(jobs, u_config)

    expected_scales = {
        0: 1.0,
        50: 2.0,
        100: 3.0,
        125: 3.5,
        150: 4.0,
        600: 4.0,
    }
    for depth, expected in expected_scales.items():
        actual = depth_noise_scale(depth)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(
                f"Noise scale mismatch at D={depth}: {actual} != {expected}."
            )

    expected_cap = {
        "p_1q": 0.000980000,
        "p_2q": 0.027880000,
        "gamma_1q": 0.000448692,
        "gamma_2q": 0.008452596,
    }
    cap_parameters = depth_noise_parameters(150)
    late_parameters = depth_noise_parameters(10_000)
    for name, expected in expected_cap.items():
        for label, values in (
            ("cap", cap_parameters),
            ("post-cap", late_parameters),
        ):
            if not math.isclose(
                values[name], expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise AssertionError(
                    f"{label} {name} mismatch: {values[name]} != {expected}."
                )

    if len(jobs) != 672:
        raise AssertionError(f"Expected 672 jobs, found {len(jobs)}.")

    fake_qubits = sorted({
        job.num_qubits for job in jobs if job.profile_name == "fake_quebec"
    })
    if fake_qubits != [2]:
        raise AssertionError(f"FakeQuebec qubit set is {fake_qubits}, expected [2].")

    for job in jobs:
        expected_folds = (1, 3, 5) if job.num_qubits == 5 else (1, 3, 5, 7)
        if job.fold_factors != expected_folds:
            raise AssertionError(
                f"Unexpected folds for {job.num_qubits}q job {job.index}: "
                f"{job.fold_factors}."
            )

    crz_2q_depths = sorted({
        job.depth
        for job in jobs
        if job.unitary == "hea_crz" and job.num_qubits == 2
    })
    if crz_2q_depths != [30, 60, 90, 120]:
        raise AssertionError(f"Unexpected 2q CRZ depths: {crz_2q_depths}.")

    if u_config.get_unitary_max_depth("random_type", 5) < 128:
        raise AssertionError("5q random_type does not reach depth 128.")

    print("PASS: Python configuration and complete-component depths validated.")
    print("PASS: capped noise scale is 3x at D=100 and 4x from D=150 onward.")
    print("PASS: maximum noise parameters:", cap_parameters)
    print("PASS: FakeQuebec is restricted to 2q jobs.")
    print("PASS: 5q uses folds [1, 3, 5]; all other q use [1, 3, 5, 7].")
    print("PASS: 2q hea_crz depths are [30, 60, 90, 120].")
    print(f"PASS: generated {len(jobs)} unique jobs.")
    print("Jobs by qubit:", dict(sorted(Counter(j.num_qubits for j in jobs).items())))
    print("Jobs by profile:", dict(Counter(j.profile_name for j in jobs)))
    if warnings:
        print("Warnings:")
        for warning in dict.fromkeys(warnings):
            print(" -", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
