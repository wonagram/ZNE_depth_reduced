"""Hardware-inspired Pauli baselines, exponential scaling and Bell-gate noise.

Keep unitary_tools.py unchanged, so the historical experiments and noise models
are preserved when both new options are off.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from unitary_tools import (
    NoiseController,
    _original_circuit_gate_depth,
    depth_noise_parameters,
)


# Reference scales only: they are not complete noise-calibrated hardware models.
# Eagle r3: historical experiment initials (adjusted for the original study).
# Heron r3: published representative SX / CZ gate-error magnitudes.
# Nighthawk r2: IBM Quantum ibm_phoenix SX=2.114E-4, CZ=1.720E-3,
#   publicly displayed on 2026-11-09 (snapshot; median isolated RB errors).
# The numbers below are used as phenomenological TOTAL Pauli-error probabilities;
# they are NOT a conversion to a physically calibrated full device noise model.
PAULI_HARDWARE_REFERENCE_PROFILES = {
    "eagle_r3": {"p_1q": 0.000245, "p_2q": 0.006970},
    "heron_r3": {"p_1q": 0.0001587, "p_2q": 0.001391},
    "nighthawk_r2": {"p_1q": 0.0002114, "p_2q": 0.001720},
}


def get_pauli_initial_probabilities(noise_model: str) -> tuple[float, float]:
    """Return baseline total stochastic-Pauli probabilities (not RB infidelities)."""
    if noise_model not in PAULI_HARDWARE_REFERENCE_PROFILES:
        raise ValueError(
            f"Unknown hardware-inspired noise model {noise_model!r}; "
            f"available: {tuple(PAULI_HARDWARE_REFERENCE_PROFILES)}"
        )
    cfg = PAULI_HARDWARE_REFERENCE_PROFILES[noise_model]
    return float(cfg["p_1q"]), float(cfg["p_2q"])


def original_u_wire_depths(components, num_qubits: int):
    """Original U-only structural depth on each logical wire, before/after G_i.

    Expands the exact selected component decomposition, including cut macros.
    Does not include input preparation, Bell preparation/measurement, reset,
    conditional corrections or artificial noise.  Disjoint gates stay parallel.
    """
    from unitary_tools import component_plan
    depths = [0] * int(num_qubits)
    before, after = [], []
    for component in components:
        before.append(tuple(depths))
        for op in component_plan(
            component, inverse=False, dress_cx=False, include_noise=False
        ):
            wires = list(op.qubits)
            if not wires:
                continue
            current = max(depths[k] for k in wires) + 1
            for k in wires:
                depths[k] = current
        after.append(tuple(depths))
    return before, after


class ExtendedNoiseController(NoiseController):
    """Backwards-compatible noise controller with two independent extensions.

    For BOTH Bell ON and Bell OFF, noise on U gates uses the SAME historical
    physical-wire structural depth and seeded Pauli RNG. Adding noise after
    Bell operations must not silently rescale noise on U gates.
    Bell-gate noise ALONE uses an explicitly supplied unitary-only prefix
    depth; Bell operations, resets and readout never advance that Bell clock.
    Constant Pauli mode freezes all error probabilities at D_fixed.
    """

    def __init__(self, *, exponential_random_pauli: bool = False,
                 constant_seeded_random_pauli: bool = False,
                 constant_pauli_depth: int = 75,
                 bell_gate_noise: bool = False, **kwargs):
        self.exponential_random_pauli = bool(exponential_random_pauli)
        self.constant_seeded_random_pauli = bool(constant_seeded_random_pauli)
        self.constant_pauli_depth = int(constant_pauli_depth)
        self.bell_gate_noise = bool(bell_gate_noise)
        if self.constant_pauli_depth < 0:
            raise ValueError('constant_pauli_depth must be nonnegative.')
        if self.constant_seeded_random_pauli and self.exponential_random_pauli:
            raise ValueError('Choose exactly one linear/exponential/constant Pauli scaling model.')
        if self.constant_seeded_random_pauli and not kwargs.get('use_increasing_random_pauli', False):
            raise ValueError('Constant seeded Pauli requires random-Pauli channels enabled.')
        if self.exponential_random_pauli and not kwargs.get('use_increasing_random_pauli', False):
            raise ValueError('Exponential scaling requires increasing random Pauli noise.')
        if self.bell_gate_noise and not (
            kwargs.get('use_increasing_random_pauli', False)
            or kwargs.get('use_increasing_amplitude_damping', False)
        ):
            raise ValueError('--bell-noise requires custom Pauli and/or amplitude damping.')
        super().__init__(**kwargs)
        self.bell_unitary_depth = 0
        self._clock_circuit = None
        self._clock = None
        # Separate RNG: turning on Bell noise must not shift the sampled U Pauli partitions.
        self._bell_rng = np.random.default_rng(self.base_seed + 983_071)
        self._bell_counts = {
            'preparation_1q_channels': 0,
            'preparation_2q_channels': 0,
            'measurement_1q_channels': 0,
            'measurement_2q_channels': 0,
        }
        self._bell_observed_depths = {'preparation': set(), 'measurement': set()}

    def _cap_depth(self):
        if self.maximum_noise_multiplier == 1.0:
            return 0.0
        if self.noise_multiplier <= 1.0:
            raise ValueError('Exponential scaling requires noise_multiplier > 1.')
        return (self.reference_depth * (self.maximum_noise_multiplier - 1.0)
                / (self.noise_multiplier - 1.0))

    def scale_at_depth(self, depth):
        depth = float(depth)
        if not math.isfinite(depth) or depth < 0:
            raise ValueError(f'Invalid nonnegative unitary depth: {depth!r}')
        if self.constant_seeded_random_pauli:
            # Freeze the *linear* formula at D*, even when passed another D.
            depth = float(self.constant_pauli_depth)
        if not self.exponential_random_pauli:
            return min(1.0 + (self.noise_multiplier - 1.0)
                       * depth / self.reference_depth, self.maximum_noise_multiplier)
        cap = self._cap_depth()
        if cap == 0.0:
            return 1.0
        return self.maximum_noise_multiplier ** min(depth / cap, 1.0)

    def parameters_at_depth(self, depth):
        if self.constant_seeded_random_pauli:
            # The same p1q/p2q after EVERY U gate and noisy Bell basis gate.
            depth = self.constant_pauli_depth
        if not self.exponential_random_pauli:
            return depth_noise_parameters(
                depth, reference_depth=self.reference_depth,
                noise_multiplier=self.noise_multiplier,
                maximum_noise_multiplier=self.maximum_noise_multiplier,
                p1_initial=self.p1_initial, p2_initial=self.p2_initial,
                gamma1_initial=self.gamma1_initial, gamma2_initial=self.gamma2_initial,
            )
        scale = self.scale_at_depth(depth)
        values = {
            'p_1q': self.p1_initial * scale,
            'p_2q': self.p2_initial * scale,
            'gamma_1q': self.gamma1_initial * scale,
            'gamma_2q': self.gamma2_initial * scale,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0 or not math.isfinite(value):
                raise ValueError(f'{name} at D={depth} is not a probability: {value}')
        return values

    def begin_unitary_copy(self, qc, wires: Sequence, *, base_depth: int = 0,
                           base_depths: Sequence[int] | None = None):
        """Seed U-only physical depths on one folded copy, excluding ancillas/Bells.

        For local folding, each G_i copy starts at the selected original-U prefix
        depth, so previous use of a recycled physical wire cannot contaminate D.
        """
        if not self.bell_gate_noise:
            return
        if self._clock_circuit is not qc:
            self._clock_circuit = qc
            self._clock = [0] * qc.num_qubits
        wires = list(wires)
        if base_depths is None:
            base_depths = [base_depth] * len(wires)
        if len(base_depths) != len(wires):
            raise ValueError('base_depths must correspond to the supplied U wires.')
        for wire, depth in zip(wires, base_depths):
            self._clock[qc.find_bit(wire).index] = int(depth)

    def _unitary_only_depth(self, qc, wires: Sequence) -> int:
        if self._clock_circuit is not qc or self._clock is None:
            self._clock_circuit = qc
            self._clock = [0] * qc.num_qubits
        indices = [qc.find_bit(wire).index for wire in wires]
        if not indices:
            raise ValueError('U noise location has no wires.')
        depth = max(self._clock[i] for i in indices) + 1
        for i in indices:
            self._clock[i] = depth
        return int(depth)

    def _depth_parameters(self, qc, wires: Sequence):
        # Fundamental ON/OFF fairness invariant: the existing noise locations
        # on U gates retain the exact SAME depth model when Bell noise is enabled.
        # Added Bell-channel instructions are labeled ARTIFICIAL_NOISE and are
        # ignored by _original_circuit_gate_depth, so this stays unchanged.
        # Only append_bell_noise() uses the separate U-only prefix clock.
        if not self.exponential_random_pauli and not self.constant_seeded_random_pauli:
            return super()._depth_parameters(qc, wires)
        depth = _original_circuit_gate_depth(qc, wires)
        parameters = self.parameters_at_depth(depth)
        self._runtime['last_original_gate_depth'] = int(depth)
        self._runtime['max_original_gate_depth'] = max(
            int(self._runtime['max_original_gate_depth']), int(depth))
        self._runtime['last_depth_dependent_parameters'] = dict(parameters)
        return int(depth), parameters

    def append_bell_noise(self, qc, wires: Sequence, *, unitary_depth: int, phase: str):
        """One noise location for one primitive Bell gate (or each X/Z parallel lane).

        Input wires length=1: local Pauli channel. length=2: two-qubit Pauli
        channel for CX. No extra channel after the actual classical readout.
        """
        if not self.bell_gate_noise:
            return
        if phase not in ('preparation', 'measurement'):
            raise ValueError(f'Unknown Bell phase: {phase!r}')
        wires = list(wires)
        if len(wires) not in (1, 2):
            raise ValueError('Bell gate noise requires one or two wires.')
        p = self.parameters_at_depth(unitary_depth)
        self._bell_observed_depths[phase].add(int(unitary_depth))
        old_rng = self.rng
        self.rng = self._bell_rng
        try:
            if len(wires) == 1:
                if self.use_increasing_random_pauli:
                    self._append_random_pauli_1q(qc, wires[0], p['p_1q'])
                if self.use_increasing_amplitude_damping:
                    self._append_increasing_amplitude(
                        qc, wires, p['gamma_1q'], two_qubit_location=False)
                self._bell_counts[f'{phase}_1q_channels'] += 1
            else:
                if self.use_increasing_random_pauli:
                    self._append_random_pauli_2q(qc, wires, p['p_2q'])
                if self.use_increasing_amplitude_damping:
                    self._append_increasing_amplitude(
                        qc, wires, p['gamma_2q'], two_qubit_location=True)
                self._bell_counts[f'{phase}_2q_channels'] += 1
        finally:
            self.rng = old_rng

    def runtime_report(self):
        result = super().runtime_report()
        result['scaling_model'] = (
            'constant_seeded_pauli_from_linear'
            if self.constant_seeded_random_pauli else
            'exponential_capped' if self.exponential_random_pauli else 'linear_capped'
        )
        if self.constant_seeded_random_pauli:
            fixed = self.parameters_at_depth(self.constant_pauli_depth)
            result['scaling_formula'] = (
                'p(D)=p_initial*min(1+(noise_multiplier-1)*D_fixed/'
                'reference_depth,maximum_noise_multiplier), for every D'
            )
            result['constant_reference_depth'] = int(self.constant_pauli_depth)
            result['constant_scale'] = float(self.scale_at_depth(self.constant_pauli_depth))
            result['constant_probabilities'] = {
                'p_1q': float(fixed['p_1q']),
                'p_2q': float(fixed['p_2q']),
            }
            result['parameters_at_depth_0'] = dict(fixed)
            result['parameters_at_reference_depth'] = dict(fixed)
            result['maximum_parameters'] = dict(fixed)
            result['increments_per_depth'] = {
                key: 0.0 for key in ('p_1q', 'p_2q', 'gamma_1q', 'gamma_2q')
            }
            result['depth_definition'] = (
                'Constant error parameters fixed from the pre-existing capped '
                'LINEAR Pauli scale at D_fixed; physical U depth is NOT used '
                'to change the channel probabilities.'
            )
        if self.exponential_random_pauli:
            result['scaling_formula'] = (
                's(D)=min(maximum_noise_multiplier**(D/cap_depth), '
                'maximum_noise_multiplier); cap_depth=reference_depth*'
                '(maximum_noise_multiplier-1)/(noise_multiplier-1)'
            )
            result['increments_per_depth'] = None  # The slope is not constant.
            result['parameters_at_depth_0'] = self.parameters_at_depth(0)
            result['parameters_at_reference_depth'] = self.parameters_at_depth(self.reference_depth)
        result['bell_gate_noise_enabled'] = bool(self.bell_gate_noise)
        if self.bell_gate_noise:
            result['bell_depth_definition'] = (
                'Bell gate noise ONLY: original-U-only primitive-gate prefix '
                'depth. Local G_i preparation uses the U prefix before G_i, '
                'measurement the prefix after G_i. Bell gates, initial '
                'preparation, readout, correction and reset do not advance '
                'this separate Bell clock. U-gate noise depth remains the same '
                'as when Bell noise is OFF.'
            )
            # Preserve the historical U-gate depth_definition from super(),
            # rather than incorrectly replacing it with the Bell-only rule.
        result['bell_noise_policy'] = (
            'H: 1q; CX: 2q; parallel Z on first + X on second: one noise '
            'LAYER with one 1q channel per qubit; measurement basis CX: 2q, '
            'H: 1q; not the classical measurement, conditional corrections or resets.'
        )
        result['bell_noise_channel_counts'] = dict(self._bell_counts)
        result['bell_noise_unitary_depths'] = {
            role: sorted(depths) for role, depths in self._bell_observed_depths.items()
        }
        return result
