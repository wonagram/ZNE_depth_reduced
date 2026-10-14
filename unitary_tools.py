"""Shared physical-depth decomposition, one-shot U compilation and noise policy.

This module is used by circuit folding, global depth reduction and local depth
reduction.  It deliberately compiles only operation plans generated for U.
Initial-state preparation, singlet preparation, Bell measurements and Pauli
post-corrections are never inspected or simplified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from U_config import GateSpec, SelectedComponent


@dataclass(frozen=True)
class PrimitiveOp:
    name: str
    qubits: tuple[int, ...]
    angle: Optional[float] = None
    tag: str = ""
    macro_id: str = ""


@dataclass(frozen=True)
class NoiseMarker:
    # Noise is attached to the emitted primitive gate, not to the logical macro.
    physical_gate: str
    qubits: tuple[int, ...]
    direction: int = 1
    macro_id: str = ""


PlanItem = Union[PrimitiveOp, NoiseMarker]


_SELF_INVERSE = {"h", "x", "y", "z", "cx"}


def _p(name: str, qubits: Sequence[int], angle: Optional[float] = None,
       *, tag: str = "", macro_id: str = "") -> PrimitiveOp:
    return PrimitiveOp(name, tuple(int(q) for q in qubits), angle, tag, macro_id)


def decompose_logical_gate(gate: GateSpec, *, macro_id: str = "") -> tuple[tuple[PrimitiveOp, ...], ...]:
    """Return physical-depth-one decomposition slices.

    A slice may contain several mutually disjoint one-qubit gates, but no qubit
    is acted on twice inside one slice.  Thus every returned slice has circuit
    depth one.
    """
    name = gate.name
    q = gate.qubits
    theta = gate.angle

    if name in {"rx", "ry", "rz"}:
        return ((_p(name, q, theta, tag="rotation", macro_id=macro_id),),)
    if name in {"x", "y", "z", "h", "s", "sx", "cx"}:
        tag = "h" if name == "h" else "logical"
        return ((_p(name, q, tag=tag, macro_id=macro_id),),)

    if len(q) != 2:
        raise ValueError(f"{name} requires two qubits.")
    control, target = q

    if name == "cz":
        # H(target) - CX(control,target) - H(target): physical depth 3.
        return (
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
        )

    if theta is None:
        raise ValueError(f"{name} requires an angle.")

    if name == "crz":
        # RZ(theta/2) - CX - RZ(-theta/2) - CX: physical depth 4.
        return (
            (_p("rz", (target,), theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), -theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
        )

    if name == "crx":
        # H - RZ(theta/2) - CX - RZ(-theta/2) - CX - H: depth 6.
        return (
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
            (_p("rz", (target,), theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), -theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
        )

    if name == "rzz":
        return (
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
        )

    if name == "rxx":
        # Parallel H basis layer, then CX-RZ-CX, then parallel H layer: depth 5.
        return (
            (
                _p("h", (control,), tag="basis_h", macro_id=macro_id),
                _p("h", (target,), tag="basis_h", macro_id=macro_id),
            ),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (
                _p("h", (control,), tag="basis_h", macro_id=macro_id),
                _p("h", (target,), tag="basis_h", macro_id=macro_id),
            ),
        )

    if name == "ryy":
        # Parallel RX basis layer, then CX-RZ-CX, then inverse basis layer: depth 5.
        return (
            (
                _p("rx", (control,), -math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
                _p("rx", (target,), -math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
            ),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (
                _p("rx", (control,), math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
                _p("rx", (target,), math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
            ),
        )

    raise ValueError(f"Unsupported logical gate {name!r}.")


def _merge_component_slices(component: SelectedComponent) -> list[PrimitiveOp]:
    """Merge corresponding decomposition slices of disjoint logical gates.

    This is the allowed overlap rule: decomposed gates may share a counted layer
    only if their primitive operations are disjoint, so the physical circuit depth
    of that counted layer remains exactly one.
    """
    per_gate = []
    for gate_index, gate in enumerate(component.gates):
        macro_id = f"{component.label}:{gate_index}"
        slices = decompose_logical_gate(gate, macro_id=macro_id)
        if component.included_depth > len(slices):
            raise ValueError(
                f"{component.label}: selected {component.included_depth} slices, "
                f"but {gate.name} has only {len(slices)}."
            )
        per_gate.append(slices[:component.included_depth])

    result: list[PrimitiveOp] = []
    for slice_index in range(component.included_depth):
        physical_slice: list[PrimitiveOp] = []
        occupied: set[int] = set()
        for slices in per_gate:
            for op in slices[slice_index]:
                if occupied.intersection(op.qubits):
                    raise ValueError(
                        f"{component.label}: decomposition slice {slice_index + 1} "
                        "has physical depth greater than one. Gates sharing a "
                        "qubit must be placed in different counted layers."
                    )
                occupied.update(op.qubits)
                physical_slice.append(op)
        result.extend(physical_slice)
    return result

def _inverse_primitive(op: PrimitiveOp) -> list[PrimitiveOp]:
    if op.name in _SELF_INVERSE:
        return [op]
    if op.name in {"rx", "ry", "rz"}:
        if op.angle is None:
            raise ValueError(f"Missing angle for {op.name}.")
        return [PrimitiveOp(op.name, op.qubits, -op.angle, op.tag, op.macro_id)]
    if op.name == "s":
        # S^dagger = S^3.  This keeps the emitted primitive alphabet free of Sdg.
        return [PrimitiveOp("s", op.qubits, None, op.tag, op.macro_id) for _ in range(3)]
    if op.name == "sx":
        # SX^dagger = SX^3.  This keeps the emitted alphabet free of SXdg.
        return [PrimitiveOp("sx", op.qubits, None, op.tag, op.macro_id) for _ in range(3)]
    raise ValueError(f"Cannot invert primitive {op.name!r}.")


def component_plan(component: SelectedComponent, *, inverse: bool = False,
                   dress_cx: bool = False, include_noise: bool = True,
                   noise_direction: int = 1) -> list[PlanItem]:
    """Build one component plan; a cut decomposed macro stays one component.

    Custom simulator noise is inserted after EVERY emitted decomposition
    primitive (H/RX/RY/RZ/S/SX/Pauli/CX).  The extra global Y gates used to
    dress CX in even depth-reduced blocks are method-generated corrections, not
    decomposition primitives, so they do not receive an extra custom channel.
    """
    primitives = _merge_component_slices(component)
    if inverse:
        inverted: list[PrimitiveOp] = []
        for op in reversed(primitives):
            inverted.extend(_inverse_primitive(op))
        primitives = inverted

    if dress_cx:
        dressed: list[PrimitiveOp] = []
        for op in primitives:
            if op.name == "cx":
                a, b = op.qubits
                dressed.extend([
                    _p("y", (a,), tag="global_y", macro_id=op.macro_id),
                    _p("y", (b,), tag="global_y", macro_id=op.macro_id),
                    op,
                    _p("y", (a,), tag="global_y", macro_id=op.macro_id),
                    _p("y", (b,), tag="global_y", macro_id=op.macro_id),
                ])
            else:
                dressed.append(op)
        primitives = dressed

    items: list[PlanItem] = []
    for op in primitives:
        items.append(op)
        if include_noise and op.tag != "global_y":
            # Preserve the old sign convention for inverse rotation gates.
            direction = -1 if inverse and op.name in {"rx", "ry", "rz"} else 1
            items.append(NoiseMarker(
                physical_gate=op.name,
                qubits=op.qubits,
                direction=direction,
                macro_id=op.macro_id,
            ))
    return items

def _candidate_signature(op: PrimitiveOp):
    if op.name == "y" and op.tag == "global_y":
        return ("global_y",)
    if op.name == "h":
        return ("h",)
    if (op.name == "rx" and op.tag == "basis_ryy" and op.angle is not None
            and abs(abs(op.angle) - math.pi / 2.0) <= 1e-12):
        return ("basis_rx", float(op.angle))
    return None


def _candidate_pair_cancels(first, second) -> bool:
    if first[0] != second[0]:
        return False
    if first[0] in {"global_y", "h"}:
        return True
    if first[0] == "basis_rx":
        # Only inverse +/-pi/2 pairs cancel.  Two identical RX(-pi/2) gates
        # do NOT cancel and are deliberately retained.
        return abs(first[1] + second[1]) <= 1e-12
    return False


def compile_u_plan_once(items: Sequence[PlanItem]) -> tuple[list[PlanItem], dict[str, int]]:
    """Single stack pass over one freshly generated U block.

    Noise markers never block a cancellation and are never removed.  Gates on
    other wires are ignored.  Any non-candidate unitary touching a wire closes
    that wire's cancellation stack.  Because this function consumes a fresh
    logical plan before it is appended to a QuantumCircuit, it cannot touch
    state preparation, Bell preparation, measurements or corrections.
    """
    stacks: dict[int, list[tuple[int, tuple]]] = {}
    removed: set[int] = set()
    stats = {"global_y_pairs": 0, "h_pairs": 0, "basis_rx_pairs": 0}

    for index, item in enumerate(items):
        if isinstance(item, NoiseMarker):
            continue
        signature = _candidate_signature(item)
        if signature is not None and len(item.qubits) == 1:
            wire = item.qubits[0]
            stack = stacks.setdefault(wire, [])
            if stack and _candidate_pair_cancels(stack[-1][1], signature):
                previous, previous_signature = stack.pop()
                removed.add(previous)
                removed.add(index)
                key = {
                    "global_y": "global_y_pairs",
                    "h": "h_pairs",
                    "basis_rx": "basis_rx_pairs",
                }[signature[0]]
                stats[key] += 1
            else:
                stack.append((index, signature))
            continue

        # Any other unitary operation involving a wire blocks cancellations
        # across that operation on that wire.
        for wire in item.qubits:
            stacks.pop(wire, None)

    compiled = [item for i, item in enumerate(items) if i not in removed]
    return compiled, stats


def build_unitary_block_plan(components: Sequence[SelectedComponent], *,
                             inverse: bool = False, dress_cx: bool = False,
                             compile_cancellations: bool = True) -> tuple[list[PlanItem], dict[str, int]]:
    ordered = list(reversed(components)) if inverse else list(components)
    raw: list[PlanItem] = []
    for component in ordered:
        raw.extend(component_plan(
            component,
            inverse=inverse,
            dress_cx=dress_cx,
            include_noise=True,
            noise_direction=-1 if inverse else 1,
        ))
    if compile_cancellations:
        return compile_u_plan_once(raw)
    return raw, {"global_y_pairs": 0, "h_pairs": 0, "basis_rx_pairs": 0}


def _resolve_wire_map(wires: Union[Sequence, Mapping[int, object]], logical: int):
    return wires[logical]


def append_plan(qc, wires: Union[Sequence, Mapping[int, object]],
                items: Sequence[PlanItem], *, noise_controller=None,
                use_noise: bool = True) -> None:
    """Append a precompiled logical plan to a Qiskit circuit."""
    for item in items:
        if isinstance(item, NoiseMarker):
            if use_noise and noise_controller is not None:
                physical = [_resolve_wire_map(wires, q) for q in item.qubits]
                noise_controller.append(
                    qc, physical, item.physical_gate, item.direction
                )
            continue

        physical = [_resolve_wire_map(wires, q) for q in item.qubits]
        name = item.name
        if name == "h":
            qc.h(physical[0])
        elif name == "x":
            qc.x(physical[0])
        elif name == "y":
            qc.y(physical[0])
        elif name == "z":
            qc.z(physical[0])
        elif name == "s":
            qc.s(physical[0])
        elif name == "sx":
            qc.sx(physical[0])
        elif name == "rx":
            qc.rx(item.angle, physical[0])
        elif name == "ry":
            qc.ry(item.angle, physical[0])
        elif name == "rz":
            qc.rz(item.angle, physical[0])
        elif name == "cx":
            qc.cx(physical[0], physical[1])
        else:
            raise ValueError(f"Unsupported emitted primitive {name!r}.")


def append_unitary_block(qc, wires, components: Sequence[SelectedComponent], *,
                         inverse: bool = False, dress_cx: bool = False,
                         noise_controller=None, use_noise: bool = True,
                         compile_cancellations: bool = True) -> dict[str, int]:
    """Generate, compile exactly once, and append one U or U-dagger block."""
    plan, stats = build_unitary_block_plan(
        components,
        inverse=inverse,
        dress_cx=dress_cx,
        compile_cancellations=compile_cancellations,
    )
    append_plan(
        qc, wires, plan, noise_controller=noise_controller, use_noise=use_noise
    )
    return stats


def append_local_component_copy(qc, wire_map, component: SelectedComponent, *,
                                block_index: int, noise_controller=None,
                                use_noise: bool = True) -> None:
    """Append one atomic local-folding component on one logical copy.

    No custom H/Y/RX cancellation pass is applied in the local method.  The
    complete (possibly truncated) decomposed macro component is folded as one G_i.
    """
    items = component_plan(
        component,
        inverse=False,
        dress_cx=(block_index % 2 == 1),
        include_noise=True,
        noise_direction=1,
    )
    append_plan(
        qc, wire_map, items,
        noise_controller=noise_controller,
        use_noise=use_noise,
    )


class NoiseController:
    """Insert custom simulator noise after each emitted primitive gate.

    Decomposed macros therefore receive noise after every H/RX/RY/RZ/... and
    every CX in their decomposition. CX keeps the historical two-qubit
    depolarizing rule plus independent damping on its two participants.
    """

    def __init__(self, *, use_depolarizing: bool,
                 use_amplitude_damping: bool,
                 use_phase_damping: bool,
                 use_coherent_overrotation: bool,
                 depolarizing_strength: float,
                 amplitude_damping_strength: float,
                 phase_damping_strength: float,
                 overrotation_epsilon: float):
        from qiskit_aer.noise import (
            amplitude_damping_error,
            depolarizing_error,
            phase_damping_error,
        )

        self.use_depolarizing = bool(use_depolarizing)
        self.use_amplitude_damping = bool(use_amplitude_damping)
        self.use_phase_damping = bool(use_phase_damping)
        self.use_coherent_overrotation = bool(use_coherent_overrotation)
        self.overrotation_epsilon = float(overrotation_epsilon)

        self.depolarizing_1q = (
            depolarizing_error(float(depolarizing_strength), 1).to_instruction()
            if self.use_depolarizing else None
        )
        self.depolarizing_2q = (
            depolarizing_error(float(depolarizing_strength), 2).to_instruction()
            if self.use_depolarizing else None
        )
        self.amplitude_1q = (
            amplitude_damping_error(
                float(amplitude_damping_strength), canonical_kraus=False
            ).to_instruction()
            if self.use_amplitude_damping else None
        )
        self.phase_1q = (
            phase_damping_error(
                float(phase_damping_strength), canonical_kraus=False
            ).to_instruction()
            if self.use_phase_damping else None
        )

    def _append_1q_channels(self, qc, wire) -> None:
        if self.depolarizing_1q is not None:
            qc.append(self.depolarizing_1q, [wire])
        if self.amplitude_1q is not None:
            qc.append(self.amplitude_1q, [wire])
        if self.phase_1q is not None:
            qc.append(self.phase_1q, [wire])

    def _append_coherent(self, qc, wire, physical_gate: str, direction: int) -> None:
        if not self.use_coherent_overrotation:
            return

        # Preserve the previous simulator semantics: coherent overrotation is
        # appended as a QuantumError instruction, rather than as an ordinary
        # logical gate that a transpiler might merge with neighboring gates.
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from qiskit_aer.noise import coherent_unitary_error

        epsilon = float(direction) * self.overrotation_epsilon
        x_like = {"rx", "x", "sx", "crx", "rxx"}
        y_like = {"ry", "y", "ryy", "h"}
        error_circuit = QuantumCircuit(1)
        if physical_gate in x_like:
            error_circuit.rx(epsilon, 0)
        elif physical_gate in y_like:
            error_circuit.ry(epsilon, 0)
        else:
            error_circuit.rz(epsilon, 0)

        qc.append(
            coherent_unitary_error(Operator(error_circuit)).to_instruction(),
            [wire],
        )

    def append(self, qc, wires: Sequence, physical_gate: str, direction: int = 1) -> None:
        wires = list(wires)
        if len(wires) == 1:
            self._append_1q_channels(qc, wires[0])
            self._append_coherent(qc, wires[0], physical_gate, direction)
            return
        if len(wires) != 2:
            raise ValueError("NoiseController supports one- and two-qubit primitives only.")

        if physical_gate != "cx":
            raise ValueError(
                f"Unexpected two-qubit primitive {physical_gate!r}; logical macros "
                "must be decomposed to one-qubit gates and CX before noise insertion."
            )

        # Historical CX policy, now applied to every decomposed CX as well.
        if self.depolarizing_2q is not None:
            qc.append(self.depolarizing_2q, wires)
        if self.amplitude_1q is not None:
            qc.append(self.amplitude_1q, [wires[0]])
            qc.append(self.amplitude_1q, [wires[1]])
        if self.phase_1q is not None:
            qc.append(self.phase_1q, [wires[0]])
            qc.append(self.phase_1q, [wires[1]])
        # Preserve the historical model: no coherent overrotation is attached to CX.



def make_fake_aer_backend(fake_backend_name: str, *, seed_simulator: int = 150,
                           required_qubits: int | None = None):
    """Create a noisy Aer simulator from an IBM fake-backend snapshot.

    The fake snapshot supplies coupling, basis gates, readout properties and
    device noise.  Aer performs the execution so mid-circuit measurement and
    if_test control flow can be simulated.  No custom NoiseController channels
    should be appended when this backend mode is selected.
    """
    if not isinstance(fake_backend_name, str) or not fake_backend_name.strip():
        raise ValueError("FAKE_BACKEND_NAME must be a non-empty fake-backend class name.")

    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider

    class_name = fake_backend_name.strip()
    backend_class = getattr(fake_provider, class_name, None)
    if backend_class is None:
        available = sorted(
            name for name in dir(fake_provider)
            if name.startswith("Fake") and name not in {"FakeBackendV2", "FakeProviderForBackendV2"}
        )
        raise ValueError(
            f"Unknown or unavailable FAKE_BACKEND_NAME={class_name!r}. "
            "Install a qiskit-ibm-runtime version that contains it, or choose one "
            f"of the available classes: {', '.join(available)}"
        )

    snapshot_backend = backend_class()
    if required_qubits is not None and snapshot_backend.num_qubits < int(required_qubits):
        raise ValueError(
            f"{class_name} has {snapshot_backend.num_qubits} qubits, but this run "
            f"needs at least {int(required_qubits)} physical qubits."
        )
    simulator = AerSimulator.from_backend(snapshot_backend)
    simulator.set_options(seed_simulator=int(seed_simulator))
    return snapshot_backend, simulator

# ============================================================
# Small exact matrices used by validation and the local checker
# ============================================================

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
_S = np.diag([1, 1j]).astype(complex)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


def _one_qubit_matrix(op: PrimitiveOp) -> np.ndarray:
    if op.name == "h":
        return _H
    if op.name == "x":
        return _X
    if op.name == "y":
        return _Y
    if op.name == "z":
        return _Z
    if op.name == "s":
        return _S
    if op.name == "sx":
        return _SX
    if op.name in {"rx", "ry", "rz"}:
        pauli = {"rx": _X, "ry": _Y, "rz": _Z}[op.name]
        return math.cos(op.angle / 2.0) * _I - 1j * math.sin(op.angle / 2.0) * pauli
    raise ValueError(op.name)


def _single_on_n(matrix: np.ndarray, logical: int, n: int) -> np.ndarray:
    result = np.array([[1]], dtype=complex)
    for q in reversed(range(n)):
        result = np.kron(result, matrix if q == logical else _I)
    return result


def _cx_on_n(control: int, target: int, n: int) -> np.ndarray:
    dim = 1 << n
    indices = np.arange(dim)
    destinations = indices ^ (((indices >> control) & 1) << target)
    matrix = np.zeros((dim, dim), dtype=complex)
    matrix[destinations, indices] = 1.0
    return matrix


def primitive_plan_matrix(primitives: Iterable[PrimitiveOp], n: int) -> np.ndarray:
    result = np.eye(1 << n, dtype=complex)
    for op in primitives:
        if op.name == "cx":
            matrix = _cx_on_n(op.qubits[0], op.qubits[1], n)
        else:
            matrix = _single_on_n(_one_qubit_matrix(op), op.qubits[0], n)
        result = matrix @ result
    return result


def selected_component_matrix(component: SelectedComponent, n: int) -> np.ndarray:
    primitives = [
        item for item in component_plan(
            component, inverse=False, dress_cx=False, include_noise=False
        ) if isinstance(item, PrimitiveOp)
    ]
    return primitive_plan_matrix(primitives, n)


def selected_unitary_matrix(components: Sequence[SelectedComponent], n: int) -> np.ndarray:
    result = np.eye(1 << n, dtype=complex)
    for component in components:
        result = selected_component_matrix(component, n) @ result
    return result
