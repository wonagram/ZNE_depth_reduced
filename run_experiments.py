"""Expand and sequentially execute ZNE experiment groups.

Examples
--------
Preview only (no simulations):
    python run_experiments.py --dry-run

Run according to experiment_config.py:
    python run_experiments.py --run

Run only one profile or experiment group:
    python run_experiments.py --run --profile aer_increasing_pauli \
        --experiment hea_cz_complete_repeats
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator

import experiment_config as cfg


@dataclass(frozen=True)
class Job:
    index: int
    total: int
    experiment_name: str
    profile_name: str
    method: str
    script: Path
    machine: str
    backend: str
    noise: str
    fake_backend_name: str | None
    num_qubits: int
    unitary: str
    depth: int
    report_directory: Path
    report_tag: str
    shots: int
    fold_factors: tuple[int, ...]
    base_seed: int
    gate_weight_1q: float
    gate_weight_2q: float
    max_simulator_physical_qubits: int

    @property
    def report_filename(self) -> str:
        return (
            f"{self.num_qubits}q_{self.unitary}_{self.depth}_"
            f"{self.report_tag}_{self.method}.json"
        )

    @property
    def report_path(self) -> Path:
        return self.report_directory / self.report_filename

    @property
    def slug(self) -> str:
        return (
            f"{self.index:04d}_{self.profile_name}_{self.method}_"
            f"{self.num_qubits}q_{self.unitary}_d{self.depth}"
        )


def _as_list(value: Any) -> list[Any]:
    if value == "all":
        raise ValueError("'all' needs contextual expansion and cannot be converted directly.")
    if isinstance(value, (str, int, float)):
        return [value]
    return list(value)


def _load_u_config(project_root: Path):
    path = project_root / "U_config.py"
    spec = importlib.util.spec_from_file_location("batch_u_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selected_names(value: Any, all_names: Iterable[str]) -> list[str]:
    if value == "all":
        return list(all_names)
    return [str(item) for item in _as_list(value)]


def _iter_experiment_points(experiment: dict[str, Any]) -> Iterator[tuple[int, int]]:
    mapping = experiment.get("depths_by_num_qubits")
    if mapping is not None:
        for qubits, depths in mapping.items():
            for depth in _as_list(depths):
                yield int(qubits), int(depth)
        return

    qubits_values = _as_list(experiment["num_qubits"])
    depth_values = _as_list(experiment["depth"])
    for qubits in qubits_values:
        for depth in depth_values:
            yield int(qubits), int(depth)


def _merged(common: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    result = dict(common)
    result.update(experiment)
    return result


def _build_jobs(
    project_root: Path,
    *,
    profile_filter: set[str] | None,
    experiment_filter: set[str] | None,
    method_filter: set[str] | None,
) -> list[Job]:
    method_files = dict(cfg.METHOD_FILES)
    all_methods = list(method_files)
    all_profiles = list(cfg.EXECUTION_PROFILES)

    raw_jobs: list[dict[str, Any]] = []
    for experiment in cfg.EXPERIMENTS:
        if not experiment.get("enabled", True):
            continue
        experiment_name = str(experiment["name"])
        if experiment_filter and experiment_name not in experiment_filter:
            continue

        merged = _merged(cfg.COMMON, experiment)
        methods = _selected_names(merged.get("methods", "all"), all_methods)
        profiles = _selected_names(merged.get("profiles", "all"), all_profiles)

        if method_filter:
            methods = [name for name in methods if name in method_filter]
        if profile_filter:
            profiles = [name for name in profiles if name in profile_filter]

        unknown_methods = set(methods) - set(method_files)
        unknown_profiles = set(profiles) - set(cfg.EXECUTION_PROFILES)
        if unknown_methods:
            raise ValueError(f"Unknown methods in {experiment_name}: {sorted(unknown_methods)}")
        if unknown_profiles:
            raise ValueError(f"Unknown profiles in {experiment_name}: {sorted(unknown_profiles)}")

        for num_qubits, depth in _iter_experiment_points(merged):
            for profile_name in profiles:
                profile = dict(cfg.EXECUTION_PROFILES[profile_name])
                for method in methods:
                    raw_jobs.append({
                        "experiment_name": experiment_name,
                        "profile_name": profile_name,
                        "method": method,
                        "script": project_root / method_files[method],
                        "machine": str(merged.get("machine", "gpu")),
                        "backend": str(profile["backend"]),
                        "noise": str(profile["noise"]),
                        "fake_backend_name": profile.get("fake_backend_name"),
                        "num_qubits": num_qubits,
                        "unitary": str(merged["unitary"]),
                        "depth": depth,
                        "report_directory": (
                            project_root / str(merged.get(
                                "report_directory", cfg.REPORT_DIRECTORY
                            ))
                        ).resolve(),
                        "report_tag": str(profile["report_tag"]),
                        "shots": int(merged.get("shots", 1000)),
                        "fold_factors": tuple(int(v) for v in merged.get(
                            "fold_factors", [1, 3, 5, 7]
                        )),
                        "base_seed": int(merged.get("base_seed", 150)),
                        "gate_weight_1q": float(merged.get("gate_weight_1q", 0.3)),
                        "gate_weight_2q": float(merged.get("gate_weight_2q", 0.7)),
                        "max_simulator_physical_qubits": int(merged.get(
                            "max_simulator_physical_qubits", 24
                        )),
                    })

    total = len(raw_jobs)
    return [Job(index=i + 1, total=total, **raw) for i, raw in enumerate(raw_jobs)]


def _validate_jobs(jobs: list[Job], u_config) -> list[str]:
    warnings: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for job in jobs:
        key = (
            job.profile_name, job.method, job.num_qubits,
            job.unitary, job.depth, job.report_path,
        )
        if key in seen:
            raise ValueError(f"Duplicate job generated: {key}")
        seen.add(key)

        if not job.script.is_file():
            raise FileNotFoundError(f"Main file not found: {job.script}")
        normalized = u_config.normalize_unitary_type(job.unitary)
        maximum = u_config.get_unitary_max_depth(normalized, job.num_qubits)
        if not 1 <= job.depth <= maximum:
            raise ValueError(
                f"{job.experiment_name}: {normalized}, {job.num_qubits}q, "
                f"depth {job.depth} exceeds allowed 1..{maximum}."
            )
        selected = u_config.select_unitary_components(
            normalized, job.num_qubits, job.depth
        )
        if selected[-1].included_depth != selected[-1].full_depth:
            raise ValueError(
                f"{job.experiment_name}: depth {job.depth} cuts the final component "
                f"{selected[-1].label} at "
                f"{selected[-1].included_depth}/{selected[-1].full_depth}."
            )

        if job.machine != "gpu":
            warnings.append(
                f"Job {job.index}: machine={job.machine!r}; the supplied experiment "
                "plan was intended for GPU execution."
            )
        if job.backend == "fake_backend" and job.noise != "none":
            raise ValueError("Fake backend profiles must use noise='none'.")
        if job.backend == "aer_simulator" and job.noise == "none":
            warnings.append(f"Job {job.index}: AerSimulator is configured without custom noise.")

        if job.method in {"global", "local"}:
            physical = job.num_qubits * max(job.fold_factors)
            if physical > 24:
                warnings.append(
                    f"{job.method}, {job.num_qubits}q, max fold "
                    f"{max(job.fold_factors)} uses up to {physical} physical qubits. "
                    "This can exceed practical GPU memory even though the local "
                    "construction guard is set to "
                    f"{job.max_simulator_physical_qubits}."
                )
    return warnings


def _command(job: Job) -> list[str]:
    command = [
        sys.executable,
        str(job.script),
        "--machine", job.machine,
        "--backend", job.backend,
        "--noise", job.noise,
        "--num-qubits", str(job.num_qubits),
        "--unitary", job.unitary,
        "--depth", str(job.depth),
        "--report-directory", str(job.report_directory),
        "--shots", str(job.shots),
        "--base-seed", str(job.base_seed),
        "--gate-weight-1q", str(job.gate_weight_1q),
        "--gate-weight-2q", str(job.gate_weight_2q),
        "--fold-factors", *[str(value) for value in job.fold_factors],
    ]
    if job.fake_backend_name:
        command.extend(["--fake-backend-name", str(job.fake_backend_name)])
    if job.method == "local":
        command.extend([
            "--max-simulator-physical-qubits",
            str(job.max_simulator_physical_qubits),
        ])
    return command



def _job_record(job: Job, **extra: Any) -> dict[str, Any]:
    record = asdict(job)
    record["script"] = str(job.script)
    record["report_directory"] = str(job.report_directory)
    record["fold_factors"] = list(job.fold_factors)
    record.update(extra)
    return record

def _valid_existing_report(job: Job) -> bool:
    path = job.report_path
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    execution = report.get("execution") or {}
    return (
        report.get("method") == job.method
        and int(report.get("num_qubits", -1)) == job.num_qubits
        and report.get("unitary_type") == job.unitary
        and int(report.get("depth", -1)) == job.depth
        and list(report.get("fold_factors", [])) == list(job.fold_factors)
        and execution.get("filename_tag") == job.report_tag
    )


def _run_and_tee(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if cfg.STREAM_OUTPUT:
                    print(line, end="")
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait()


def _git_checkpoint(project_root: Path, message: str) -> bool:
    options = cfg.GIT
    if not options.get("enabled", False):
        return False
    paths = [str(path) for path in options.get("paths", [])]
    if not paths:
        return False
    subprocess.run(["git", "add", "--", *paths], cwd=project_root, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=project_root,
    )
    if staged.returncode == 0:
        return False
    subprocess.run(["git", "commit", "-m", message], cwd=project_root, check=True)
    if options.get("push", False):
        command = ["git", "push", str(options.get("remote", "origin"))]
        branch = options.get("branch")
        if branch:
            command.append(str(branch))
        subprocess.run(command, cwd=project_root, check=True)
    return True


def _print_job(job: Job) -> None:
    physical = (
        job.num_qubits * max(job.fold_factors)
        if job.method in {"global", "local"}
        else job.num_qubits
    )
    print(
        f"[{job.index:04d}/{job.total:04d}] "
        f"{job.profile_name} | {job.method} | {job.num_qubits}q "
        f"{job.unitary} depth={job.depth} | estimated wires={physical}"
    )
    print(f"    report: {job.report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only.")
    mode.add_argument("--run", action="store_true", help="Execute simulations.")
    parser.add_argument("--profile", action="append", help="Run only named profile(s).")
    parser.add_argument("--experiment", action="append", help="Run only named experiment group(s).")
    parser.add_argument("--method", action="append", choices=tuple(cfg.METHOD_FILES))
    parser.add_argument("--limit", type=int, help="Limit the expanded job list after filtering.")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--no-git", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    u_config = _load_u_config(project_root)

    known_profiles = set(cfg.EXECUTION_PROFILES)
    known_experiments = {str(item["name"]) for item in cfg.EXPERIMENTS}
    if args.profile:
        unknown = set(args.profile) - known_profiles
        if unknown:
            raise ValueError(f"Unknown --profile value(s): {sorted(unknown)}")
    if args.experiment:
        unknown = set(args.experiment) - known_experiments
        if unknown:
            raise ValueError(f"Unknown --experiment value(s): {sorted(unknown)}")

    jobs = _build_jobs(
        project_root,
        profile_filter=set(args.profile) if args.profile else None,
        experiment_filter=set(args.experiment) if args.experiment else None,
        method_filter=set(args.method) if args.method else None,
    )
    if not jobs:
        raise ValueError("The selected filters generated zero jobs.")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        jobs = jobs[:args.limit]
        jobs = [Job(**{**asdict(job), "index": i + 1, "total": len(jobs)})
                for i, job in enumerate(jobs)]

    warnings = _validate_jobs(jobs, u_config)
    dry_run = cfg.DRY_RUN
    if args.dry_run:
        dry_run = True
    elif args.run:
        dry_run = False

    print("=" * 72)
    print("ZNE BATCH PLAN")
    print(f"Project root : {project_root}")
    print(f"Jobs         : {len(jobs)}")
    print(f"Mode         : {'DRY RUN (no simulation)' if dry_run else 'EXECUTE'}")
    print(f"GPU visible  : {cfg.CUDA_VISIBLE_DEVICES}")
    print("=" * 72)

    for warning in dict.fromkeys(warnings):
        print("WARNING:", warning)
    if warnings:
        print()

    if dry_run:
        for job in jobs:
            _print_job(job)
            print("    command:", " ".join(_command(job)))
        print("\nNo simulations were executed because dry-run mode is enabled.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = (project_root / cfg.LOG_DIRECTORY / timestamp).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = log_root / "batch_summary.json"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)
    env.setdefault("PYTHONUNBUFFERED", "1")

    if args.no_git:
        cfg.GIT["enabled"] = False

    results: list[dict[str, Any]] = []
    new_successes = 0
    failures = 0
    skipped = 0
    batch_start = time.time()

    for job in jobs:
        _print_job(job)
        job.report_directory.mkdir(parents=True, exist_ok=True)

        if cfg.SKIP_EXISTING and not args.rerun_existing and _valid_existing_report(job):
            print("    status: SKIP (matching report already exists)\n")
            skipped += 1
            results.append(_job_record(
                job,
                status="skipped",
                runtime_seconds=0.0,
                report_path=str(job.report_path),
            ))
            summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue

        command = _command(job)
        log_path = log_root / f"{job.slug}.log"
        start = time.time()
        returncode = _run_and_tee(command, cwd=project_root, env=env, log_path=log_path)
        runtime = time.time() - start

        report_valid = returncode == 0 and _valid_existing_report(job)
        status = "success" if report_valid else "failed"
        if report_valid:
            new_successes += 1
            print(f"    status: SUCCESS ({runtime:.1f} s)\n")
        else:
            failures += 1
            print(
                f"    status: FAILED (return code {returncode}, {runtime:.1f} s)\n"
                f"    log: {log_path}\n"
            )

        results.append(_job_record(
            job,
            status=status,
            returncode=returncode,
            runtime_seconds=runtime,
            report_path=str(job.report_path),
            log_path=str(log_path),
        ))
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        every = int(cfg.GIT.get("commit_every_successes", 0) or 0)
        if report_valid and every > 0 and new_successes % every == 0:
            _git_checkpoint(
                project_root,
                f"Add ZNE batch results through job {job.index}/{job.total}",
            )

        if failures and cfg.STOP_ON_FAILURE:
            break

    _git_checkpoint(project_root, f"Add final ZNE batch results {timestamp}")

    elapsed = time.time() - batch_start
    print("=" * 72)
    print("BATCH COMPLETE")
    print(f"Succeeded : {new_successes}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failures}")
    print(f"Elapsed   : {elapsed / 3600.0:.2f} h")
    print(f"Summary   : {summary_path}")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
