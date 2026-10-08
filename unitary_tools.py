"""Shared logical-gate decomposition, one-shot U compilation and noise policy.

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
    logical_gate: str
    qubits: tuple[int, ...]
    direction: int = 1
    macro_id: str = ""


PlanItem = Union[PrimitiveOp, NoiseMarker]


_SELF_INVERSE = {"h", "x", "y", "z", "cx"}


def _p(name: str, qubits: Sequence[int], angle: Optional[float] = None,
       *, tag: str = "", macro_id: str = "") -> PrimitiveOp:
    return PrimitiveOp(name, tuple(int(q) for q in qubits), angle, tag, macro_id)


def decompose_logical_gate(gate: GateSpec, *, macro_id: str = "") -> tuple[tuple[PrimitiveOp, ...], ...]:
    """Return counted decomposition slices exactly following the requested rules."""
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
        # H(target) - CX(control,target) - H(target), counted as depth 1.
        return ((
            _p("h", (target,), tag="basis_h", macro_id=macro_id),
            _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
            _p("h", (target,), tag="basis_h", macro_id=macro_id),
        ),)

    if theta is None:
        raise ValueError(f"{name} requires an angle.")

    if name == "crz":
        # RZ(theta/2) - CX - RZ(-theta/2) - CX, counted as depth 1.
        return ((
            _p("rz", (target,), theta / 2.0, tag="macro_rotation", macro_id=macro_id),
            _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
            _p("rz", (target,), -theta / 2.0, tag="macro_rotation", macro_id=macro_id),
            _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
        ),)

    if name == "crx":
        # H - RZ(theta/2) - CX - RZ(-theta/2) - CX - H, depth 1.
        return ((
            _p("h", (target,), tag="basis_h", macro_id=macro_id),
            _p("rz", (target,), theta / 2.0, tag="macro_rotation", macro_id=macro_id),
            _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
            _p("rz", (target,), -theta / 2.0, tag="macro_rotation", macro_id=macro_id),
            _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
            _p("h", (target,), tag="basis_h", macro_id=macro_id),
        ),)

    if name == "rzz":
        return (
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
        )

    if name == "rxx":
        # Put both H gates in the same counted slice as the adjacent CX.
        return (
            (
                _p("h", (control,), tag="basis_h", macro_id=macro_id),
                _p("h", (target,), tag="basis_h", macro_id=macro_id),
                _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
            ),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (
                _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
                _p("h", (control,), tag="basis_h", macro_id=macro_id),
                _p("h", (target,), tag="basis_h", macro_id=macro_id),
            ),
        )

    if name == "ryy":
        # Put both +/-pi/2 RX basis changes in the CX slices.
        return (
            (
                _p("rx", (control,), -math.pi / 2.0,
                   tag="basis_ryy", macro_id=macro_id),
                _p("rx", (target,), -math.pi / 2.0,
                   tag="basis_ryy", macro_id=macro_id),
                _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
            ),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (
                _p("cx", (control, target), tag="macro_cx", macro_id=macro_id),
                _p("rx", (control,), math.pi / 2.0,
                   tag="basis_ryy", macro_id=macro_id),
                _p("rx", (target,), math.pi / 2.0,
                   tag="basis_ryy", macro_id=macro_id),
            ),
        )

    raise ValueError(f"Unsupported logical gate {name!r}.")


def _merge_component_slices(component: SelectedComponent) -> list[PrimitiveOp]:
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
        micro_sequences = [list(slices[slice_index]) for slices in per_gate]
        # Interleave microsteps of disjoint gates so matching gates remain
        # visually and structurally parallel whenever possible.
        for micro_index in range(max(len(seq) for seq in micro_sequences)):
            for seq in micro_sequences:
                if micro_index < len(seq):
                    result.append(seq[micro_index])
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
    """Build one component plan; a cut depth-three gate remains one component."""
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

    items: list[PlanItem] = list(primitives)
    if include_noise:
        direction = -1 if inverse else int(noise_direction)
        for gate_index, gate in enumerate(component.gates):
            items.append(NoiseMarker(
                logical_gate=gate.name,
                qubits=gate.qubits,
                direction=direction,
                macro_id=f"{component.label}:{gate_index}",
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
                    qc, physical, item.logical_gate, item.direction
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
    complete (possibly truncated) macro component is folded as one G_i.
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
    """Insert exactly one configured noise boundary per logical gate.

    Legacy RX/RY/RZ/CX gates preserve the old policy.  A new decomposed two-
    qubit macro receives one independent one-qubit channel on each participant,
    only after the whole macro (or selected cut fragment) finishes.
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

    def _append_coherent(self, qc, wire, logical_gate: str, direction: int) -> None:
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
        if logical_gate in x_like:
            error_circuit.rx(epsilon, 0)
        elif logical_gate in y_like:
            error_circuit.ry(epsilon, 0)
        else:
            error_circuit.rz(epsilon, 0)

        qc.append(
            coherent_unitary_error(Operator(error_circuit)).to_instruction(),
            [wire],
        )

    def append(self, qc, wires: Sequence, logical_gate: str, direction: int = 1) -> None:
        wires = list(wires)
        if len(wires) == 1:
            self._append_1q_channels(qc, wires[0])
            self._append_coherent(qc, wires[0], logical_gate, direction)
            return
        if len(wires) != 2:
            raise ValueError("NoiseController supports one- and two-qubit gates only.")

        if logical_gate == "cx":
            # Preserve the original random_type CX noise rule exactly.
            if self.depolarizing_2q is not None:
                qc.append(self.depolarizing_2q, wires)
            if self.amplitude_1q is not None:
                qc.append(self.amplitude_1q, [wires[0]])
                qc.append(self.amplitude_1q, [wires[1]])
            if self.phase_1q is not None:
                qc.append(self.phase_1q, [wires[0]])
                qc.append(self.phase_1q, [wires[1]])
            # The historical code did not attach coherent error to CX.
            return

        # New decomposed two-qubit gates: one channel per qubit, once, after
        # the whole macro or selected fragment.
        for wire in wires:
            self._append_1q_channels(qc, wire)
            self._append_coherent(qc, wire, logical_gate, direction)


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
