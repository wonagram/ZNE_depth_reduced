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
import signal
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
    depth_round: int
    batch_phase: int
    phase_label: str

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


def _iter_experiment_points(
    experiment: dict[str, Any],
) -> Iterator[tuple[int, int, int]]:
    """Yield (num_qubits, depth, depth_round) in depth-major order."""
    mapping = experiment.get("depths_by_num_qubits")
    if mapping is not None:
        normalized = {
            int(qubits): [int(depth) for depth in _as_list(depths)]
            for qubits, depths in mapping.items()
        }
        max_rounds = max((len(depths) for depths in normalized.values()), default=0)
        for depth_round in range(max_rounds):
            for qubits in sorted(normalized):
                depths = normalized[qubits]
                if depth_round < len(depths):
                    yield qubits, depths[depth_round], depth_round
        return

    qubits_values = sorted(int(value) for value in _as_list(experiment["num_qubits"]))
    depth_values = [int(value) for value in _as_list(experiment["depth"])]
    for depth_round, depth in enumerate(depth_values):
        for qubits in qubits_values:
            yield qubits, depth, depth_round


def _merged(common: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    result = dict(common)
    result.update(experiment)
    return result


def _value_for_num_qubits(
    settings: dict[str, Any],
    key: str,
    num_qubits: int,
    default: Any,
) -> Any:
    """Return a per-qubit override such as fold_factors_by_num_qubits.

    Integer and string dictionary keys are both accepted so the configuration
    remains easy to edit or load from JSON-like sources.
    """
    mapping = settings.get(f"{key}_by_num_qubits")
    if mapping is None:
        return default
    if not isinstance(mapping, dict):
        raise TypeError(f"{key}_by_num_qubits must be a dictionary.")
    if num_qubits in mapping:
        return mapping[num_qubits]
    string_key = str(num_qubits)
    if string_key in mapping:
        return mapping[string_key]
    return default


def _profile_allows_num_qubits(profile: dict[str, Any], num_qubits: int) -> bool:
    allowed = profile.get("allowed_num_qubits")
    if allowed is None:
        return True
    return int(num_qubits) in {int(value) for value in _as_list(allowed)}


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
    method_order = {name: index for index, name in enumerate(all_methods)}
    profile_default_order = {
        name: index for index, name in enumerate(all_profiles)
    }

    raw_jobs: list[dict[str, Any]] = []
    for experiment_order, experiment in enumerate(cfg.EXPERIMENTS):
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

        for num_qubits, depth, depth_round in _iter_experiment_points(merged):
            fold_factor_values = _value_for_num_qubits(
                merged,
                "fold_factors",
                num_qubits,
                merged.get("fold_factors", [1, 3, 5, 7]),
            )
            fold_factors = tuple(int(value) for value in _as_list(fold_factor_values))

            for profile_name in profiles:
                profile = dict(cfg.EXECUTION_PROFILES[profile_name])
                if not _profile_allows_num_qubits(profile, num_qubits):
                    continue
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
                        "fold_factors": fold_factors,
                        "base_seed": int(merged.get("base_seed", 150)),
                        "gate_weight_1q": float(merged.get("gate_weight_1q", 0.3)),
                        "gate_weight_2q": float(merged.get("gate_weight_2q", 0.7)),
                        "max_simulator_physical_qubits": int(merged.get(
                            "max_simulator_physical_qubits", 24
                        )),
                        "depth_round": int(depth_round),
                        "batch_phase": int(profile.get("batch_phase", 0)),
                        "phase_label": str(profile.get(
                            "phase_label",
                            f"Batch phase {int(profile.get('batch_phase', 0))}",
                        )),
                        "_sort_key": (
                            int(profile.get("batch_phase", 0)),
                            int(depth_round),
                            # Finish the low-cost circuit-folding jobs before
                            # global/local jobs in the same depth round.
                            int(method_order[method]),
                            # Within global/local, prefer fewer estimated wires.
                            # With 5q folds [1,3,5], 25-wire 5q jobs run before
                            # 28-wire 4q jobs, reducing the chance that one heavy
                            # 4q job blocks all 5q results.
                            int(
                                num_qubits * max(fold_factors)
                                if method in {"global", "local"}
                                else num_qubits
                            ),
                            int(num_qubits),
                            int(experiment_order),
                            int(profile.get(
                                "order_priority",
                                profile_default_order[profile_name],
                            )),
                        ),
                    })

    raw_jobs.sort(key=lambda item: item["_sort_key"])
    for raw in raw_jobs:
        raw.pop("_sort_key", None)

    total = len(raw_jobs)
    return [Job(index=i + 1, total=total, **raw) for i, raw in enumerate(raw_jobs)]



def _stage_key(job: Job) -> tuple[int, int]:
    """Return the strict batch-stage key: phase first, then depth round."""
    return int(job.batch_phase), int(job.depth_round)


def _group_jobs_by_stage(jobs: list[Job]) -> list[tuple[tuple[int, int], list[Job]]]:
    """Group already-sorted jobs without changing their execution order."""
    groups: list[tuple[tuple[int, int], list[Job]]] = []
    for job in jobs:
        key = _stage_key(job)
        if not groups or groups[-1][0] != key:
            groups.append((key, [job]))
        else:
            groups[-1][1].append(job)
    return groups


def _validate_strict_batch_order(jobs: list[Job]) -> None:
    """Reject any plan that can return to an earlier depth round or phase.

    The required sequence is:
      phase -1: small FakeQuebec smoke test;
      phase  0: all normal-Aer jobs, globally depth-major;
      phase  1: all full FakeQuebec jobs, globally depth-major.
    """
    previous: tuple[int, int] | None = None
    for job in jobs:
        current = _stage_key(job)
        if previous is not None and current < previous:
            raise ValueError(
                "Batch ordering regression detected: "
                f"job {job.index} has phase/round {current} after {previous}. "
                "A later depth round must never be followed by an earlier one."
            )
        previous = current

        if job.batch_phase == -1:
            if job.profile_name != "fake_quebec_statevector_smoke":
                raise ValueError("Only the dedicated smoke profile may use batch_phase=-1.")
        elif job.batch_phase == 0:
            if job.backend in {"fake_backend", "fake_statevector"}:
                raise ValueError(
                    "The normal-Aer phase must exclude FakeQuebec completely."
                )
        elif job.batch_phase == 1:
            if job.profile_name != "fake_quebec_statevector":
                raise ValueError(
                    "The final phase must contain only full FakeQuebec jobs."
                )
        else:
            raise ValueError(f"Unsupported batch_phase={job.batch_phase}.")


def _stage_banner(stage_jobs: list[Job]) -> str:
    """Human-readable phase/round banner for dry runs and actual execution."""
    first = stage_jobs[0]
    depths = sorted({job.depth for job in stage_jobs})
    profiles = sorted({job.profile_name for job in stage_jobs})
    depth_text = ", ".join(str(value) for value in depths)
    profile_text = ", ".join(profiles)

    if first.batch_phase == -1:
        round_text = "smoke test"
    else:
        round_text = f"depth round {first.depth_round + 1}"

    return (
        "\n" + "#" * 96 + "\n"
        f"# {first.phase_label} | {round_text}\n"
        f"# configured depths in this round: [{depth_text}]\n"
        f"# profiles: {profile_text}\n"
        f"# jobs in this section: {len(stage_jobs)}\n"
        + "#" * 96
    )


def _validate_jobs(jobs: list[Job], u_config) -> list[str]:
    _validate_strict_batch_order(jobs)
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
        if not job.fold_factors:
            raise ValueError(f"Job {job.index}: fold_factors must not be empty.")
        if any(value < 1 or value % 2 == 0 for value in job.fold_factors):
            raise ValueError(
                f"Job {job.index}: fold_factors must be positive odd integers."
            )
        if tuple(sorted(set(job.fold_factors))) != job.fold_factors:
            raise ValueError(
                f"Job {job.index}: fold_factors must be strictly increasing and unique."
            )

        profile = dict(cfg.EXECUTION_PROFILES[job.profile_name])
        if not _profile_allows_num_qubits(profile, job.num_qubits):
            raise ValueError(
                f"Job {job.index}: profile {job.profile_name!r} does not allow "
                f"{job.num_qubits} qubits."
            )
        if job.backend in {"fake_backend", "fake_statevector"} and job.noise != "none":
            raise ValueError(
                "Fake-backend profiles must use noise='none'; the snapshot "
                "already supplies its device noise model."
            )
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



def _format_duration(seconds: float) -> str:
    """Format elapsed time as HH:MM:SS, or Nd HH:MM:SS for long batches."""
    total = max(0, int(round(seconds)))
    days, remainder = divmod(total, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def _terminate_process_group(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Terminate the child and any process it started."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_child(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    stream_output: bool,
    progress_seconds: float,
    timeout_seconds: float | None,
    heartbeat,
) -> int:
    """Run one main file, always saving full output to its per-job log.

    In quiet batch mode, child output is not copied to the terminal. A concise
    heartbeat is printed periodically instead. In stream mode, the historical
    behavior is retained and every child-output line is also printed.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()

        if stream_output:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            except KeyboardInterrupt:
                _terminate_process_group(process)
                raise
            return process.wait()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        start = time.monotonic()
        poll_interval = progress_seconds if progress_seconds > 0 else 30.0
        try:
            while True:
                elapsed = time.monotonic() - start
                if timeout_seconds is not None and elapsed >= timeout_seconds:
                    log.write(
                        f"\nBATCH TIMEOUT: exceeded {timeout_seconds:.0f} seconds; "
                        "terminating process group.\n"
                    )
                    log.flush()
                    _terminate_process_group(process)
                    return 124
                wait_for = poll_interval
                if timeout_seconds is not None:
                    wait_for = min(wait_for, max(0.1, timeout_seconds - elapsed))
                try:
                    return process.wait(timeout=wait_for)
                except subprocess.TimeoutExpired:
                    if progress_seconds > 0:
                        heartbeat(time.monotonic() - start)
        except KeyboardInterrupt:
            _terminate_process_group(process)
            raise


def _preflight_environment(jobs: list[Job], project_root: Path) -> None:
    """Fail once before the batch if common imports or GPU Aer are unavailable."""
    gpu_required = any(
        job.machine.lower() == "gpu" and job.backend != "real_hardware"
        for job in jobs
    )
    code_lines = [
        "import numpy",
        "import scipy",
        "import matplotlib.pyplot",
        "import qiskit",
        "import qiskit_aer",
        "import qiskit_ibm_runtime",
        "from qiskit_aer import AerSimulator",
    ]
    if gpu_required:
        code_lines.extend([
            "devices = tuple(AerSimulator().available_devices())",
            "assert 'GPU' in devices, f'Aer GPU is unavailable; devices={devices}'",
        ])
    code_lines.append("print('environment imports and GPU check passed')")

    completed = subprocess.run(
        [sys.executable, "-c", "; ".join(code_lines)],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        output = completed.stdout.strip()
        raise RuntimeError(
            "Batch environment preflight failed before job 1.\n"
            "Install/fix the missing dependency, then rerun.\n\n"
            + output
        )


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



def _job_description(job: Job) -> str:
    physical = (
        job.num_qubits * max(job.fold_factors)
        if job.method in {"global", "local"}
        else job.num_qubits
    )
    return (
        f"{job.profile_name} | {job.method} | "
        f"{job.num_qubits}q {job.unitary} depth={job.depth} | wires~{physical}"
    )


def _progress_counts(successes: int, failures: int, skipped: int) -> str:
    return f"S={successes} F={failures} K={skipped}"


def _print_dry_job(job: Job) -> None:
    print(
        f"[{job.index:04d}/{job.total:04d}] PLAN    | "
        f"round={job.depth_round + 1} | {_job_description(job)}"
    )


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
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Also print every main-file output line. Full output is always saved in logs.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        help="Override the quiet-mode heartbeat interval from experiment_config.py.",
    )
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="In dry-run mode, print the full command for each job.",
    )
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

    stream_output = bool(getattr(cfg, "STREAM_OUTPUT", False) or args.stream_output)
    progress_seconds = (
        float(args.progress_seconds)
        if args.progress_seconds is not None
        else float(getattr(cfg, "PROGRESS_UPDATE_SECONDS", 30))
    )
    if progress_seconds < 0:
        raise ValueError("--progress-seconds must be zero or positive.")
    configured_timeout = getattr(cfg, "JOB_TIMEOUT_SECONDS", None)
    timeout_seconds = (
        None
        if configured_timeout in (None, 0, 0.0)
        else float(configured_timeout)
    )
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("JOB_TIMEOUT_SECONDS must be positive, 0, or None.")

    print("=" * 88)
    print("ZNE BATCH PLAN")
    print(f"Project root    : {project_root}")
    print(f"Jobs            : {len(jobs)}")
    print(f"Mode            : {'DRY RUN (no simulation)' if dry_run else 'EXECUTE'}")
    print(f"Console output  : {'FULL CHILD OUTPUT' if stream_output else 'CONCISE PROGRESS'}")
    if not stream_output:
        print(f"Heartbeat       : every {progress_seconds:g} s (0 disables it)")
    print(f"GPU visible     : {cfg.CUDA_VISIBLE_DEVICES}")
    print(
        "Job timeout     : "
        + ("disabled" if timeout_seconds is None else _format_duration(timeout_seconds))
    )
    print(
        "Order guarantee : ALL normal-Aer round 1 -> round 2 -> "
        "round 3 -> round 4/5 -> ALL FakeQuebec rounds"
    )
    print("=" * 88)

    for warning in dict.fromkeys(warnings):
        print("WARNING:", warning)
    if warnings:
        print()

    if dry_run:
        for _, stage_jobs in _group_jobs_by_stage(jobs):
            print(_stage_banner(stage_jobs))
            for job in stage_jobs:
                _print_dry_job(job)
                if args.show_commands:
                    print("    command:", " ".join(_command(job)))
        print("\nNo simulations were executed because dry-run mode is enabled.")
        if not args.show_commands:
            print("Use --show-commands only when you need to inspect full commands.")
        return 0

    print("Preflight       : checking imports and GPU Aer ...", flush=True)
    _preflight_environment(jobs, project_root)
    print("Preflight       : PASS\n", flush=True)

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
    batch_start = time.monotonic()
    stage_groups = _group_jobs_by_stage(jobs)
    stage_jobs_by_key = {key: members for key, members in stage_groups}
    previous_stage: tuple[int, int] | None = None

    for job in jobs:
        current_stage = _stage_key(job)
        if current_stage != previous_stage:
            print(_stage_banner(stage_jobs_by_key[current_stage]), flush=True)
            previous_stage = current_stage

        job.report_directory.mkdir(parents=True, exist_ok=True)
        batch_elapsed = time.monotonic() - batch_start

        if cfg.SKIP_EXISTING and not args.rerun_existing and _valid_existing_report(job):
            skipped += 1
            print(
                f"[{job.index:04d}/{job.total:04d}] SKIP    | {_job_description(job)} | "
                f"batch={_format_duration(batch_elapsed)} | "
                f"{_progress_counts(new_successes, failures, skipped)}",
                flush=True,
            )
            results.append(_job_record(
                job,
                status="skipped",
                runtime_seconds=0.0,
                report_path=str(job.report_path),
            ))
            summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue

        print(
            f"[{job.index:04d}/{job.total:04d}] START   | {_job_description(job)} | "
            f"batch={_format_duration(batch_elapsed)} | "
            f"{_progress_counts(new_successes, failures, skipped)}",
            flush=True,
        )

        command = _command(job)
        log_path = log_root / f"{job.slug}.log"
        start = time.monotonic()

        def heartbeat(job_elapsed: float) -> None:
            print(
                f"[{job.index:04d}/{job.total:04d}] RUNNING | "
                f"job={_format_duration(job_elapsed)} | "
                f"batch={_format_duration(time.monotonic() - batch_start)} | "
                f"{_progress_counts(new_successes, failures, skipped)}",
                flush=True,
            )

        returncode = _run_child(
            command,
            cwd=project_root,
            env=env,
            log_path=log_path,
            stream_output=stream_output,
            progress_seconds=progress_seconds,
            timeout_seconds=timeout_seconds,
            heartbeat=heartbeat,
        )
        runtime = time.monotonic() - start

        report_valid = returncode == 0 and _valid_existing_report(job)
        timed_out = returncode == 124
        status = "success" if report_valid else "timeout" if timed_out else "failed"
        if report_valid:
            new_successes += 1
            print(
                f"[{job.index:04d}/{job.total:04d}] SUCCESS | {_job_description(job)} | "
                f"job={_format_duration(runtime)} | "
                f"batch={_format_duration(time.monotonic() - batch_start)} | "
                f"{_progress_counts(new_successes, failures, skipped)}",
                flush=True,
            )
        else:
            failures += 1
            failure_label = "TIMEOUT" if timed_out else "FAILED "
            print(
                f"[{job.index:04d}/{job.total:04d}] {failure_label} | {_job_description(job)} | "
                f"rc={returncode} | job={_format_duration(runtime)} | "
                f"batch={_format_duration(time.monotonic() - batch_start)} | "
                f"{_progress_counts(new_successes, failures, skipped)}",
                flush=True,
            )
            print(f"    log: {log_path}", flush=True)

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
            print("STOP_ON_FAILURE=True: stopping after the first failure.", flush=True)
            break

    _git_checkpoint(project_root, f"Add final ZNE batch results {timestamp}")

    elapsed = time.monotonic() - batch_start
    processed = new_successes + failures + skipped
    print("\n" + "=" * 88)
    print("BATCH COMPLETE")
    print(f"Processed       : {processed}/{len(jobs)}")
    print(f"Succeeded       : {new_successes}")
    print(f"Skipped         : {skipped}")
    print(f"Failed          : {failures}")
    print(f"Elapsed         : {_format_duration(elapsed)}")
    print(f"Summary         : {summary_path}")
    print(f"Per-job logs    : {log_root}")
    print("=" * 88)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
