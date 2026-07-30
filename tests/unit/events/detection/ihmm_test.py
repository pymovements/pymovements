# Copyright (c) 2026 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Tests functionality of the IHMM algorithm."""
import copy

import numpy as np
import polars as pl
import pytest

import pymovements as pm
from pymovements.events.detection.idt import idt
from pymovements.events.detection.ihmm import _baum_backward
from pymovements.events.detection.ihmm import _baum_forward
from pymovements.events.detection.ihmm import _baum_welch
from pymovements.events.detection.ihmm import _collapse_states
from pymovements.events.detection.ihmm import _compute_hmm
from pymovements.events.detection.ihmm import _emit_log_prob
from pymovements.events.detection.ihmm import _format_optimal_dict
from pymovements.events.detection.ihmm import _log_sum_exp
from pymovements.events.detection.ihmm import _viterbi
from pymovements.events.detection.ihmm import ihmm
from pymovements.synthetic import step_function
from pymovements.transforms.numpy import pos2vel


# -----------------------------------------------------------------------------
# _emit_log_prob
# -----------------------------------------------------------------------------


def test_emit_log_prob_matches_closed_form_gaussian():
    """The implementation should match the analytical Gaussian log-PDF."""
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 2.0])

    v = 0.5
    s = 0

    expected = (
        -0.5 * np.log(2 * np.pi * sigma[s] ** 2)
        - ((v - mu[s]) ** 2) / (2 * sigma[s] ** 2)
    )

    result = _emit_log_prob(mu=mu, sigma=sigma, v=v, s=s)

    assert np.isclose(result, expected, atol=1e-12)


def test_emit_log_prob_uses_correct_state_parameters():
    """Different states should yield different probabilities."""
    mu = np.array([0.0, 100.0])
    sigma = np.array([1.0, 1.0])

    v = 0.0

    state0 = _emit_log_prob(mu=mu, sigma=sigma, v=v, s=0)
    state1 = _emit_log_prob(mu=mu, sigma=sigma, v=v, s=1)

    assert state0 > state1


def test_emit_log_prob_sigma_floor_prevents_instability():
    """Very small sigma should not produce NaN or inf."""
    mu = np.array([0.0, 0.0])
    sigma = np.array([0.0, 1.0])

    result = _emit_log_prob(mu=mu, sigma=sigma, v=0.0, s=0)

    assert np.isfinite(result)


@pytest.mark.parametrize(
    ('mu', 'sigma'),
    [
        pytest.param(None, np.array([1.0, 1.0]), id='mu_none'),
        pytest.param(np.array([0.0, 0.0]), None, id='sigma_none'),
        pytest.param(None, None, id='both_none'),
    ],
)
def test_emit_log_prob_raises_on_none_parameters(mu, sigma):
    """Mu and sigma are required; None must raise instead of being silently ignored."""
    with pytest.raises(ValueError, match='mu and sigma must not be None'):
        _emit_log_prob(mu=mu, sigma=sigma, v=0.0, s=0)


# -----------------------------------------------------------------------------
# _log_sum_exp
# -----------------------------------------------------------------------------


def test_log_sum_exp_matches_manual_computation():
    arr = np.array([-2.0, -1.0, -0.5])

    expected = np.log(np.sum(np.exp(arr)))

    result = _log_sum_exp(arr)

    assert np.isclose(result, expected, atol=1e-12)


def test_log_sum_exp_is_numerically_stable():
    """Large negative values should still produce finite output."""
    arr = np.array([-1000.0, -1001.0, -1002.0])

    result = _log_sum_exp(arr)

    expected = -1000.0 + np.log(
        1 + np.exp(-1.0) + np.exp(-2.0),
    )

    assert np.isfinite(result)
    assert np.isclose(result, expected, atol=1e-10)


# -----------------------------------------------------------------------------
# _format_optimal_dict
# -----------------------------------------------------------------------------


def test_format_optimal_dict_converts_to_json_serializable():
    """Should convert numpy arrays to lists of floats and exponentiate log probs."""
    opt = {
        'mu': np.array([1.0, 2.0]),
        'sigma': np.array([0.5, 0.5]),
        'init': np.log(np.array([0.7, 0.3])),
        'trans': np.log(np.array([[0.9, 0.1], [0.2, 0.8]])),
    }

    result = _format_optimal_dict(opt)

    assert result['mu'] == [1.0, 2.0]
    assert result['sigma'] == [0.5, 0.5]
    assert np.isclose(result['init'][0], 0.7)
    assert np.isclose(result['init'][1], 0.3)
    assert np.isclose(result['trans'][0][0], 0.9)
    assert np.isclose(result['trans'][0][1], 0.1)
    assert np.isclose(result['trans'][1][0], 0.2)
    assert np.isclose(result['trans'][1][1], 0.8)


# -----------------------------------------------------------------------------
# Forward / Backward consistency
# -----------------------------------------------------------------------------


def hmm_parameters():
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
    velocities = np.array([0.1, -0.2, 9.9, 10.2])
    mask = np.array([True, True, True, True])
    return {
        'mu': mu,
        'sigma': sigma,
        'init': init,
        'trans': trans,
        'velocities': velocities,
        'mask': mask,
    }


def test_baum_forward_shape():
    params = hmm_parameters()

    alpha = _baum_forward(
        mu=params['mu'],
        sigma=params['sigma'],
        init=params['init'],
        trans=params['trans'],
        velocities=params['velocities'],
        velocities_mask=params['mask'],
        T=len(params['velocities']),
        M=2,
    )

    assert alpha.shape == (4, 2)


def test_baum_backward_shape():
    params = hmm_parameters()

    beta = _baum_backward(
        mu=params['mu'],
        sigma=params['sigma'],
        trans=params['trans'],
        velocities=params['velocities'],
        velocities_mask=params['mask'],
        T=len(params['velocities']),
        M=2,
    )

    assert beta.shape == (4, 2)


def test_forward_backward_produce_same_log_likelihood():
    """Forward and backward algorithms must agree on sequence likelihood."""
    params = hmm_parameters()

    alpha = _baum_forward(
        mu=params['mu'],
        sigma=params['sigma'],
        init=params['init'],
        trans=params['trans'],
        velocities=params['velocities'],
        velocities_mask=params['mask'],
        T=len(params['velocities']),
        M=2,
    )

    beta = _baum_backward(
        mu=params['mu'],
        sigma=params['sigma'],
        trans=params['trans'],
        velocities=params['velocities'],
        velocities_mask=params['mask'],
        T=len(params['velocities']),
        M=2,
    )

    forward_ll = _log_sum_exp(alpha[-1])

    backward_terms = []
    for s in range(2):
        backward_terms.append(
            params['init'][s]
            + _emit_log_prob(
                mu=params['mu'],
                sigma=params['sigma'],
                v=params['velocities'][0],
                s=s,
            )
            + beta[0, s],
        )

    backward_ll = _log_sum_exp(np.array(backward_terms))

    assert np.isclose(forward_ll, backward_ll, atol=1e-10)


def test_forward_handles_masked_values():
    """Masked observations should skip emission contribution."""
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([0.0, np.nan, 10.0])
    mask = np.array([True, False, True])

    alpha = _baum_forward(
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
        T=3,
        M=2,
    )

    assert np.all(np.isfinite(alpha))


def test_baum_forward_raises_on_none_mu_or_sigma():
    """Mu/sigma are required for the forward pass; None must raise, not be ignored."""
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    velocities = np.array([0.0, 1.0, 2.0])
    mask = np.array([True, True, True])

    with pytest.raises(ValueError, match='mu and sigma must not be None'):
        _baum_forward(
            mu=None,
            sigma=None,
            init=init,
            trans=trans,
            velocities=velocities,
            velocities_mask=mask,
            T=3,
            M=2,
        )


def test_baum_forward_raises_on_none_mu_or_sigma_even_if_fully_masked():
    """The None check must not depend on velocities_mask reaching _emit_log_prob."""
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    velocities = np.array([0.0, 1.0, 2.0])
    mask = np.array([False, False, False])

    with pytest.raises(ValueError, match='mu and sigma must not be None'):
        _baum_forward(
            mu=None,
            sigma=None,
            init=init,
            trans=trans,
            velocities=velocities,
            velocities_mask=mask,
            T=3,
            M=2,
        )


def test_baum_backward_raises_on_none_mu_or_sigma():
    """Mu/sigma are required for the backward pass; None must raise, not be ignored."""
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    velocities = np.array([0.0, 1.0, 2.0])
    mask = np.array([True, True, True])

    with pytest.raises(ValueError, match='mu and sigma must not be None'):
        _baum_backward(
            mu=None,
            sigma=None,
            trans=trans,
            velocities=velocities,
            velocities_mask=mask,
            T=3,
            M=2,
        )


# -----------------------------------------------------------------------------
# Viterbi
# -----------------------------------------------------------------------------


def test_viterbi_prefers_low_velocity_state():
    """Low velocities should map to the low-mean state."""
    mu = np.array([0.0, 20.0])
    sigma = np.array([1.0, 1.0])

    init = np.log(np.array([0.5, 0.5]))

    trans = np.log(
        np.array(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ],
        ),
    )

    velocities = np.array([0.1, -0.1, 0.0, 0.2])
    mask = np.array([True, True, True, True])

    states = _viterbi(
        states=2,
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
    )

    expected = np.array([0, 0, 0, 0])

    np.testing.assert_array_equal(states, expected)


def test_viterbi_detects_state_transition():
    """Sequence with distinct low/high velocities should transition states."""
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])

    init = np.log(np.array([0.5, 0.5]))

    trans = np.log(
        np.array(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ],
        ),
    )

    velocities = np.array([0.0, 0.1, 9.8, 10.2])
    mask = np.array([True, True, True, True])

    states = _viterbi(
        states=2,
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
    )

    expected = np.array([0, 0, 1, 1])

    np.testing.assert_array_equal(states, expected)


def test_viterbi_raises_on_none_mu_or_sigma():
    """Mu/sigma are required for decoding; None must raise, not be ignored."""
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.95, 0.05], [0.05, 0.95]]))
    velocities = np.array([0.0, 0.1, 9.8, 10.2])
    mask = np.array([True, True, True, True])

    with pytest.raises(ValueError, match='mu and sigma must not be None'):
        _viterbi(
            states=2,
            mu=None,
            sigma=None,
            init=init,
            trans=trans,
            velocities=velocities,
            velocities_mask=mask,
        )


def test_viterbi_skips_emission_for_masked_first_observation():
    """A masked first sample must not feed its (nan) value into the emission term."""
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.7, 0.3]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([np.nan, 0.1, 10.0])
    mask = np.array([False, True, True])  # first observation is masked

    path = _viterbi(
        states=2,
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
    )

    assert path.shape == (3,)
    assert set(np.unique(path)).issubset({0, 1})


# -----------------------------------------------------------------------------
# _collapse_states
# -----------------------------------------------------------------------------


def test_collapse_states_extracts_fixation_segments():
    states = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    timesteps = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    onsets, offsets = _collapse_states(states, timesteps)

    np.testing.assert_array_equal(onsets, np.array([1, 4]))
    np.testing.assert_array_equal(offsets, np.array([2, 6]))


def test_collapse_states_handles_full_fixation_sequence():
    states = np.array([0, 0, 0, 0])
    timesteps = np.array([0, 1, 2, 3])

    onsets, offsets = _collapse_states(states, timesteps)

    np.testing.assert_array_equal(onsets, np.array([0]))
    np.testing.assert_array_equal(offsets, np.array([3]))


def test_collapse_states_handles_no_fixations():
    states = np.array([1, 1, 1, 1])
    timesteps = np.array([0, 1, 2, 3])

    onsets, offsets = _collapse_states(states, timesteps)

    assert len(onsets) == 0
    assert len(offsets) == 0


def test_collapse_states_respects_min_duration():
    """Fixations shorter than min_duration should be filtered out."""
    states = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])

    timesteps = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])

    onsets, offsets = _collapse_states(states, timesteps, min_duration=20)

    np.testing.assert_array_equal(onsets, np.array([40]))
    np.testing.assert_array_equal(offsets, np.array([60]))


def test_collapse_states_handles_empty_input():
    """Empty arrays should return empty arrays."""
    onsets, offsets = _collapse_states(np.array([]), np.array([]))

    assert len(onsets) == 0
    assert len(offsets) == 0


# -----------------------------------------------------------------------------
# Baum-Welch reestimation
# -----------------------------------------------------------------------------


def test_baum_welch_returns_valid_shapes():
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])

    init = np.log(np.array([0.5, 0.5]))

    trans = np.log(
        np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ],
        ),
    )

    velocities = np.array([0.1, 0.0, 0.2, 10.0, 9.9, 10.2])
    mask = np.array([True] * len(velocities))

    result = _baum_welch(
        states=2,
        mu=mu.copy(),
        sigma=sigma.copy(),
        init=init.copy(),
        trans=trans.copy(),
        velocities=velocities,
        velocities_mask=mask,
        max_iters=10,
    )

    assert result['mu'].shape == (2,)
    assert result['sigma'].shape == (2,)
    assert result['init'].shape == (2,)
    assert result['trans'].shape == (2, 2)


def test_baum_welch_transition_rows_sum_to_one_in_probability_space():
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])

    init = np.log(np.array([0.5, 0.5]))

    trans = np.log(
        np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
            ],
        ),
    )

    velocities = np.array([0.0, 0.1, 10.0, 10.1])
    mask = np.array([True] * len(velocities))

    result = _baum_welch(
        states=2,
        mu=mu.copy(),
        sigma=sigma.copy(),
        init=init.copy(),
        trans=trans.copy(),
        velocities=velocities,
        velocities_mask=mask,
        max_iters=5,
    )

    trans_probs = np.exp(result['trans'])

    row_sums = trans_probs.sum(axis=1)

    assert np.allclose(row_sums, np.array([1.0, 1.0]), atol=1e-6)


def test_baum_welch_updates_means_toward_observed_clusters():
    """Estimated means should separate low/high velocity clusters."""
    velocities = np.array(
        [
            0.0,
            0.1,
            -0.1,
            0.2,
            10.0,
            10.2,
            9.8,
            10.1,
        ],
    )

    mask = np.array([True] * len(velocities))

    result = _baum_welch(
        states=2,
        mu=np.array([2.0, 8.0]),
        sigma=np.array([3.0, 3.0]),
        init=np.log(np.array([0.5, 0.5])),
        trans=np.log(np.array([[0.9, 0.1], [0.1, 0.9]])),
        velocities=velocities,
        velocities_mask=mask,
        max_iters=50,
    )

    estimated_mu = np.sort(result['mu'])

    assert estimated_mu[0] < 2.0
    assert estimated_mu[1] > 8.0


@pytest.mark.parametrize(
    'none_param',
    ['mu', 'sigma', 'init', 'trans'],
)
def test_baum_welch_raises_on_none_parameters(none_param):
    """Mu, sigma, init, trans are required; None must raise, not silently (re-)initialize."""
    params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0]),
        'init': np.log(np.array([0.5, 0.5])),
        'trans': np.log(np.array([[0.9, 0.1], [0.1, 0.9]])),
    }
    params[none_param] = None

    velocities = np.array([0.0, 0.1, 10.0, 10.1])
    mask = np.array([True] * len(velocities))

    with pytest.raises(ValueError, match='mu, sigma, init and trans must not be None'):
        _baum_welch(
            states=2,
            velocities=velocities,
            velocities_mask=mask,
            max_iters=5,
            **params,
        )


def test_baum_welch_does_not_mutate_input_parameters():
    """Baum-Welch must not modify the caller's mu, sigma, init or trans arrays in place."""
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    mu_before = mu.copy()
    sigma_before = sigma.copy()
    init_before = init.copy()
    trans_before = trans.copy()

    velocities = np.array([0.1, 0.0, 0.2, 10.0, 9.9, 10.2])
    mask = np.array([True] * len(velocities))

    _baum_welch(
        states=2,
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
        max_iters=10,
    )

    np.testing.assert_array_equal(mu, mu_before)
    np.testing.assert_array_equal(sigma, sigma_before)
    np.testing.assert_array_equal(init, init_before)
    np.testing.assert_array_equal(trans, trans_before)


# -----------------------------------------------------------------------------
# _compute_hmm
# -----------------------------------------------------------------------------


def test_compute_hmm_returns_one_state_per_observation():
    velocities = np.array([0.0, 0.1, 10.0, 10.1])
    mask = np.array([True, True, True, True])

    states = _compute_hmm(
        velocities=velocities,
        verbose=False,
        reestimation=False,
        reestimation_max_iters=10,
        mu=None,
        sigma=None,
        init_state=None,
        transition_probabilities=None,
        velocities_mask=mask,
        hmm_parameters_dict=None,
    )

    assert states.shape == (4,)


def test_compute_hmm_returns_binary_states_only():
    velocities = np.array([0.0, 0.1, 10.0, 10.1])
    mask = np.array([True, True, True, True])

    states = _compute_hmm(
        velocities=velocities,
        verbose=False,
        reestimation=False,
        reestimation_max_iters=10,
        mu=None,
        sigma=None,
        init_state=None,
        transition_probabilities=None,
        velocities_mask=mask,
        hmm_parameters_dict=None,
    )

    assert set(np.unique(states)).issubset({0, 1})


def test_compute_hmm_accepts_custom_parameters():
    """Should accept and use custom HMM parameters."""
    velocities = np.array([0.0, 0.1, 10.0, 10.1])
    mask = np.array([True, True, True, True])

    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init_state = np.array([0.5, 0.5])
    transition_probabilities = np.array([[0.95, 0.05], [0.05, 0.95]])

    states = _compute_hmm(
        velocities=velocities,
        verbose=False,
        reestimation=False,
        reestimation_max_iters=10,
        mu=mu,
        sigma=sigma,
        init_state=init_state,
        transition_probabilities=transition_probabilities,
        velocities_mask=mask,
        hmm_parameters_dict=None,
    )

    assert states.shape == (4,)
    assert set(np.unique(states)).issubset({0, 1})


# -----------------------------------------------------------------------------
# ihmm integration tests
# -----------------------------------------------------------------------------


def test_ihmm_detects_fixation_event_on_synthetic_data():
    """Low velocity segment should be classified as fixation."""
    velocities = np.array(
        [
            [10.0, 10.0],
            [10.0, 10.0],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.0],
            [10.0, 10.0],
        ],
    )

    events = ihmm(
        velocities=velocities,
        minimum_duration=1,
    )

    assert len(events.frame) >= 1

    first_event = events.frame.row(0)

    onset = first_event[1]
    offset = first_event[2]

    assert onset <= offset


def test_ihmm_accepts_integer_timesteps():
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
        ],
    )

    timesteps = np.array([0, 1, 2])

    events = ihmm(
        velocities=velocities,
        timesteps=timesteps,
        minimum_duration=1,
    )

    assert events is not None


def test_ihmm_accepts_float_timesteps():
    """Float timesteps should be accepted and flow through to onsets, matching ivt."""
    positions = step_function(
        length=200,
        steps=[1],
        values=[(50., 50.)],
        start_value=(0., 0.),
    )
    velocities = pos2vel(positions, sampling_rate=sampling_rate)
    timesteps = np.arange(len(velocities), dtype=float) * 0.5

    events = ihmm(velocities, timesteps=timesteps, minimum_duration=1, reestimation=True)

    assert len(events.frame) >= 1
    assert events.frame['onset'].dtype == pl.Float64


def test_ihmm_accepts_polars_series_timesteps():
    """A polars Series of timesteps should be accepted, matching ivt."""
    positions = step_function(
        length=200,
        steps=[1],
        values=[(50., 50.)],
        start_value=(0., 0.),
    )
    velocities = pos2vel(positions, sampling_rate=sampling_rate)
    timesteps = pl.Series('time', np.arange(len(velocities)))

    events = ihmm(velocities, timesteps=timesteps, minimum_duration=1, reestimation=True)

    assert len(events.frame) >= 1


def test_ihmm_rejects_non_numeric_polars_timesteps():
    """A polars Series with a non-numeric dtype must raise TypeError, matching ivt."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])
    timesteps = pl.Series(['a', 'b'])

    with pytest.raises(TypeError, match='timesteps dtype must be float or int but is String'):
        ihmm(
            velocities=velocities,
            timesteps=timesteps,
            minimum_duration=1,
        )


def test_ihmm_rejects_invalid_mu_shape():
    velocities = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
    )

    with pytest.raises(ValueError, match='mu'):
        ihmm(
            velocities=velocities,
            mu=[1.0, 2.0, 3.0],
            minimum_duration=1,
        )


def test_ihmm_rejects_invalid_transition_shape():
    velocities = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
    )

    with pytest.raises(ValueError, match='transition_probabilities'):
        ihmm(
            velocities=velocities,
            transition_probabilities=[[0.5, 0.5]],
            minimum_duration=1,
        )


@pytest.mark.parametrize(
    'bad_transition_probabilities',
    [
        pytest.param([[0.75, 0.75], [0.05, 0.95]], id='one_row_sums_above_one'),
        pytest.param([[0.3, 0.3], [0.2, 0.2]], id='both_rows_sum_below_one'),
        pytest.param([[0.95, 0.05], [1.1, 0.2]], id='one_row_sums_above_one_other_valid'),
    ],
)
def test_ihmm_rejects_transition_probabilities_not_summing_to_one(
        bad_transition_probabilities,
):
    """Every row of transition_probabilities must sum to one, regardless of the other row."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match='transition_probabilities values must sum up to one'):
        ihmm(
            velocities=velocities,
            transition_probabilities=bad_transition_probabilities,
            minimum_duration=1,
        )


def test_ihmm_accepts_transition_probabilities_summing_to_one():
    """A valid row-stochastic transition matrix must not be rejected."""
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ],
    )

    events = ihmm(
        velocities=velocities,
        transition_probabilities=[[0.95, 0.05], [0.05, 0.95]],
        minimum_duration=1,
    )

    assert events is not None


def test_ihmm_rejects_hmm_parameters_dict_transition_not_summing_to_one():
    """The trans matrix inside hmm_parameters_dict must be validated the same way."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0]),
        'init': np.array([0.5, 0.5]),
        'trans': np.array([[0.75, 0.75], [0.05, 0.95]]),
    }

    with pytest.raises(ValueError, match='transition_probabilities values must sum up to one'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_rejects_init_state_not_summing_to_one():
    """init_state must be a probability distribution that sums to one."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match='init_state values must sum up to one'):
        ihmm(
            velocities=velocities,
            init_state=[0.3, 0.3],
            minimum_duration=1,
        )


def test_ihmm_rejects_hmm_parameters_dict_init_not_summing_to_one():
    """The init vector inside hmm_parameters_dict must be validated the same way."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0]),
        'init': np.array([0.3, 0.3]),
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
    }

    with pytest.raises(ValueError, match='init_state values must sum up to one'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_handles_nan_velocities():
    """An interior NaN sample must not crash detection and must not split the fixation."""
    velocities = np.array(
        [
            [0.0, 0.0],
            [np.nan, np.nan],
            [0.0, 0.0],
        ],
    )

    events = ihmm(
        velocities=velocities,
        minimum_duration=1,
    )

    # the low-velocity segment spans the interior NaN and yields a single fixation.
    assert len(events.frame) == 1
    onset = events.frame['onset'].to_list()[0]
    offset = events.frame['offset'].to_list()[0]
    assert onset == 0
    assert offset == 2


def test_ihmm_returns_no_events_for_all_nan_velocities():
    """A recording without a single valid sample must yield no events, not crash."""
    velocities = np.full((300, 2), np.nan)

    events = ihmm(velocities=velocities, minimum_duration=1)

    assert len(events.frame) == 0


def test_ihmm_returns_no_events_for_empty_velocities():
    """Empty input must yield no events, not crash on the initialization."""
    events = ihmm(velocities=np.empty((0, 2)), minimum_duration=1)

    assert len(events.frame) == 0


def test_ihmm_rejects_non_integer_minimum_duration():
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(TypeError, match='minimum_duration must be of type int'):
        ihmm(
            velocities=velocities,
            minimum_duration=1.5,
        )


def test_ihmm_rejects_zero_or_negative_minimum_duration():
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match='minimum_duration must be greater than 0'):
        ihmm(
            velocities=velocities,
            minimum_duration=0,
        )


def test_ihmm_rejects_negative_fractional_minimum_duration_as_type_error():
    """A negative non-integer must fail the type check first, not the value check."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(TypeError, match='minimum_duration must be of type int'):
        ihmm(
            velocities=velocities,
            minimum_duration=-1.5,
        )


def test_ihmm_accepts_numpy_integer_minimum_duration():
    """A numpy integer should be accepted just like a Python int."""
    positions = step_function(
        length=200,
        steps=[1],
        values=[(50., 50.)],
        start_value=(0., 0.),
    )
    velocities = pos2vel(positions, sampling_rate=sampling_rate)

    events = ihmm(velocities, minimum_duration=np.int64(50), reestimation=True)

    assert len(events.frame) == 1


def test_ihmm_state_order_independent_of_parameter_order_without_reestimation():
    """The same HMM given fixation-first or saccade-first must yield identical events."""
    positions = step_function(
        length=200,
        steps=[1],
        values=[(50., 50.)],
        start_value=(0., 0.),
    )
    velocities = pos2vel(positions, sampling_rate=sampling_rate)

    fixation_first = ihmm(
        velocities,
        mu=[5.0, 100.0],
        sigma=[5.0, 50.0],
        init_state=[0.9, 0.1],
        transition_probabilities=[[0.95, 0.05], [0.05, 0.95]],
        minimum_duration=1,
    )
    # Same two states, relabeled: high-velocity state first.
    saccade_first = ihmm(
        velocities,
        mu=[100.0, 5.0],
        sigma=[50.0, 5.0],
        init_state=[0.1, 0.9],
        transition_probabilities=[[0.95, 0.05], [0.05, 0.95]],
        minimum_duration=1,
    )

    assert len(fixation_first.frame) >= 1
    assert fixation_first.frame['onset'].to_list() == saccade_first.frame['onset'].to_list()
    assert fixation_first.frame['offset'].to_list() == saccade_first.frame['offset'].to_list()


def test_ihmm_accepts_hmm_parameters_dict():
    """Should accept and use hmm_parameters_dict."""
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ],
    )

    hmm_params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0]),
        'init': np.array([0.5, 0.5]),
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
    }

    events = ihmm(
        velocities=velocities,
        hmm_parameters_dict=hmm_params,
        minimum_duration=1,
    )

    assert events is not None


def test_ihmm_accepts_hmm_parameters_dict_with_keys_in_any_order():
    """The dict keys are unordered; a valid dict must not be rejected for key order."""
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ],
    )

    # correct keys, but a different insertion order than ['mu', 'sigma', 'init', 'trans'].
    hmm_params = {
        'sigma': np.array([1.0, 1.0]),
        'mu': np.array([0.0, 10.0]),
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
        'init': np.array([0.5, 0.5]),
    }

    events = ihmm(
        velocities=velocities,
        hmm_parameters_dict=hmm_params,
        minimum_duration=1,
    )

    assert events is not None


def test_ihmm_does_not_mutate_hmm_parameters_dict():
    """Ihmm must not convert or overwrite the caller's hmm_parameters_dict values."""
    velocities = step_function(
        length=200,
        steps=[100],
        values=[(9.0, 9.0)],
        start_value=(0.0, 0.0),
    )
    velocities = pos2vel(velocities)

    hmm_params = {
        'mu': [2.0, 69.0],
        'sigma': [1.3, 87.0],
        'init': [0.9, 0.1],
        'trans': [[0.97, 0.03], [0.07, 0.93]],
    }
    hmm_params_before = copy.deepcopy(hmm_params)

    ihmm(
        velocities=velocities,
        hmm_parameters_dict=hmm_params,
        reestimation=True,
        reestimation_max_iters=5,
        minimum_duration=1,
    )

    assert hmm_params == hmm_params_before


def test_ihmm_rejects_invalid_hmm_parameters_dict_keys():
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0]),
        'wrong_key': np.array([0.5, 0.5]),
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
    }

    with pytest.raises(ValueError, match='hmm_parameters_dict'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_removes_leading_trailing_nans():
    """Leading and trailing NaN velocities should be trimmed."""
    velocities = np.array(
        [
            [np.nan, np.nan],
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [np.nan, np.nan],
        ],
    )

    events = ihmm(
        velocities=velocities,
        minimum_duration=1,
    )

    assert events is not None


def test_ihmm_runs_with_reestimation_enabled():
    """Should run successfully with reestimation enabled."""
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ],
    )

    events = ihmm(
        velocities=velocities,
        reestimation=True,
        reestimation_max_iters=5,
        minimum_duration=1,
    )

    assert events is not None


# -----------------------------------------------------------------------------
# Deterministic mathematical regression tests
# -----------------------------------------------------------------------------


def test_known_gaussian_log_probability_regression():
    """Regression test with analytically verified expected value."""
    mu = np.array([0.0, 1.0])
    sigma = np.array([1.0, 1.0])

    result = _emit_log_prob(
        mu=mu,
        sigma=sigma,
        v=0.0,
        s=0,
    )

    expected = -0.9189385332046727

    assert np.isclose(result, expected, atol=1e-12)


def test_log_sum_exp_regression_value():
    arr = np.log(np.array([1.0, 2.0, 3.0]))

    result = _log_sum_exp(arr)

    expected = np.log(6.0)

    assert np.isclose(result, expected, atol=1e-12)


def test_viterbi_regression_known_path():
    """Known sequence should produce deterministic decoding."""
    mu = np.array([0.0, 5.0])
    sigma = np.array([0.5, 0.5])

    init = np.log(np.array([0.9, 0.1]))

    trans = np.log(
        np.array(
            [
                [0.95, 0.05],
                [0.10, 0.90],
            ],
        ),
    )

    velocities = np.array([0.0, 0.1, 5.2, 5.0, 0.0])
    mask = np.array([True] * len(velocities))

    result = _viterbi(
        states=2,
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
    )

    expected = np.array([0, 0, 1, 1, 0])

    np.testing.assert_array_equal(result, expected)


# -----------------------------------------------------------------------------
# _format_optimal_dict regression
# -----------------------------------------------------------------------------


def test_format_optimal_dict_regression():
    """Regression test for _format_optimal_dict output structure."""
    opt = {
        'mu': np.array([2.0, 8.0]),
        'sigma': np.array([1.5, 2.5]),
        'init': np.log(np.array([0.6, 0.4])),
        'trans': np.log(np.array([[0.85, 0.15], [0.25, 0.75]])),
    }

    result = _format_optimal_dict(opt)

    expected_keys = {'mu', 'sigma', 'init', 'trans'}
    assert set(result.keys()) == expected_keys

    assert len(result['mu']) == 2
    assert len(result['sigma']) == 2
    assert len(result['init']) == 2
    assert len(result['trans']) == 2
    assert len(result['trans'][0]) == 2
    assert len(result['trans'][1]) == 2

    # All values should be floats
    assert all(isinstance(x, float) for x in result['mu'])
    assert all(isinstance(x, float) for x in result['sigma'])
    assert all(isinstance(x, float) for x in result['init'])
    assert all(isinstance(x, float) for row in result['trans'] for x in row)


sampling_rate = 1000.0


def _spans(events):
    """Return a list of (onset, offset) tuples for every detected event."""
    frame = events.frame
    return list(zip(frame['onset'].to_list(), frame['offset'].to_list()))


def test_ihmm_detects_no_fixations_for_saccade_only_signal():
    """A signal that is all high-velocity saccades should yield no fixations."""
    positions_no_fix = step_function(
        length=6,
        steps=[1, 2, 3, 4, 5],
        values=[
            (100., 100.),
            (-100., -100.),
            (100., -100.),
            (-100., 100.),
            (200., 0.),
        ],
        start_value=(0., 0.),
    )

    velocities_no_fix = pos2vel(positions_no_fix, sampling_rate=sampling_rate)

    ihmm_events = ihmm(velocities_no_fix, reestimation=True, name='ihmm_fix')
    idt_events = idt(positions_no_fix, name='idt_fix')

    assert len(ihmm_events.frame) == 0
    assert len(idt_events.frame) == 0


def test_ihmm_detects_single_fixation():
    """A signal that settles on one location should yield a single fixation."""
    length_one_fix = 200
    positions_one_fix = step_function(
        length=length_one_fix,
        steps=[1],
        values=[(50., 50.)],
        start_value=(0., 0.),
    )

    velocities_one_fix = pos2vel(positions_one_fix, sampling_rate=sampling_rate)

    ihmm_events = ihmm(velocities_one_fix, reestimation=True, name='ihmm_fix')
    idt_events = idt(positions_one_fix, name='idt_fix')

    assert len(ihmm_events.frame) == 1
    assert len(idt_events.frame) == 1


def test_ihmm_detects_fixation_spanning_whole_recording():
    """A constant-position signal should yield one fixation covering the recording."""
    length_big_fix = 200
    positions_big_fix = np.tile(np.array([[75., 75.]]), (length_big_fix, 1))

    velocities_big_fix = pos2vel(positions_big_fix, sampling_rate=sampling_rate)

    ihmm_events = ihmm(velocities_big_fix, reestimation=True, name='ihmm_fix')
    idt_events = idt(positions_big_fix, name='idt_fix')

    assert len(ihmm_events.frame) == 1
    assert len(idt_events.frame) == 1

    # The single fixation should cover (nearly) the whole recording.
    min_span = length_big_fix * 0.9
    for events in (ihmm_events, idt_events):
        onset, offset = _spans(events)[0]
        assert offset - onset >= min_span


def test_ihmm_detects_fixations_at_recording_start_and_end():
    """A signal with a fixation before and after a single saccade should yield two events."""
    length_edges = 200
    split_at = 90
    positions_edges = step_function(
        length=length_edges,
        steps=[split_at],
        values=[(60., 60.)],
        start_value=(10., 10.),
    )

    velocities_edges = pos2vel(positions_edges, sampling_rate=sampling_rate)

    ihmm_events = ihmm(velocities_edges, reestimation=True, name='ihmm_fix', minimum_duration=2)
    idt_events = idt(positions_edges, name='idt_fix', minimum_duration=2)

    assert len(ihmm_events.frame) == 2
    assert len(idt_events.frame) == 2

    for events in (ihmm_events, idt_events):
        event_spans = sorted(_spans(events))
        first_onset = event_spans[0][0]
        last_offset = event_spans[-1][1]
        # first fixation starts at the trial start (1 sample of slack, ms at 1000 Hz).
        assert first_onset <= 1
        # last fixation ends at the trial end.
        assert last_offset >= (length_edges - 1) * (1000.0 / sampling_rate) - 1


@pytest.mark.network
def test_ihmm_fixation_count_comparable_to_idt_on_toy_dataset(tmp_path):
    """On real recordings, ihmm should detect a fixation count close to idt's."""
    dataset = pm.Dataset('ToyDataset', path=tmp_path)
    dataset.download()
    dataset.load()

    dataset.pix2deg()
    dataset.pos2vel()

    toy_gaze = dataset.gaze[0]
    toy_positions = toy_gaze.samples.select('position').to_series().to_list()
    toy_velocities = toy_gaze.samples.select('velocity').to_series().to_list()

    toy_velocities = [[np.nan, np.nan] if v == [None, None] else v for v in toy_velocities]

    ihmm_events = ihmm(toy_velocities, reestimation=True, name='ihmm_fix')
    idt_events = idt(np.array(toy_positions), name='idt_fix')

    ihmm_count, idt_count = len(ihmm_events.frame), len(idt_events.frame)
    larger = max(ihmm_count, idt_count)
    smaller = max(min(ihmm_count, idt_count), 1)
    relative_diff = (larger - smaller) / smaller
    # allow up to 50% relative disagreement on real, noisy data.
    assert relative_diff <= 0.5


def test_ihmm_detects_expected_fixation_count_on_synthetic_scanpath():
    """On a synthetic multi-fixation scanpath, ihmm should recover the fixation count."""
    rng = np.random.default_rng(seed=42)

    fixation_targets = [
        (0., 0.), (120., 5.), (240., -10.), (360., 15.), (480., 0.),
        (600., 20.), (720., -15.), (840., 10.), (960., 0.),
    ]
    fixation_len = 150  # samples held per fixation target

    length_realistic = fixation_len * len(fixation_targets)
    steps_realistic = [fixation_len * i for i in range(1, len(fixation_targets))]

    positions_realistic = step_function(
        length=length_realistic,
        steps=steps_realistic,
        values=fixation_targets[1:],
        start_value=fixation_targets[0],
    )

    jitter = rng.normal(loc=0.0, scale=0.05, size=positions_realistic.shape)
    positions_realistic = positions_realistic + jitter

    velocities_realistic = pos2vel(positions_realistic, sampling_rate=sampling_rate)

    ihmm_events = ihmm(velocities_realistic, reestimation=True, name='ihmm_fix')
    idt_events = idt(positions_realistic, name='idt_fix')

    expected_fixations = len(fixation_targets)

    # allow off-by-one at the boundaries.
    for events in (ihmm_events, idt_events):
        assert abs(len(events.frame) - expected_fixations) <= 1


# ---

def test_ihmm_rejects_hmm_parameters_dict_with_invalid_mu_shape():
    """hmm_parameters_dict with invalid mu shape should raise ValueError."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': np.array([0.0, 10.0, 5.0]),  # shape (3,) instead of (2,)
        'sigma': np.array([1.0, 1.0]),
        'init': np.array([0.5, 0.5]),
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
    }

    with pytest.raises(ValueError, match='mu.*must have shape'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_rejects_hmm_parameters_dict_with_invalid_sigma_shape():
    """hmm_parameters_dict with invalid sigma shape should raise ValueError."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0, 2.0]),  # shape (3,) instead of (2,)
        'init': np.array([0.5, 0.5]),
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
    }

    with pytest.raises(ValueError, match='sigma.*must have shape'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_rejects_hmm_parameters_dict_with_invalid_init_shape():
    """hmm_parameters_dict with invalid init shape should raise ValueError."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0]),
        'init': np.array([0.5, 0.5, 0.0]),  # shape (3,) instead of (2,)
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
    }

    with pytest.raises(ValueError, match='init_state.*must have shape'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_rejects_hmm_parameters_dict_with_invalid_trans_shape():
    """hmm_parameters_dict with invalid trans shape should raise ValueError."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': np.array([0.0, 10.0]),
        'sigma': np.array([1.0, 1.0]),
        'init': np.array([0.5, 0.5]),
        # shape (3, 2) instead of (2, 2)
        'trans': np.array([[0.95, 0.05], [0.05, 0.95], [0.1, 0.9]]),
    }

    with pytest.raises(ValueError, match='transition_probabilities.*must have shape'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_rejects_hmm_parameters_dict_with_mu_none_and_invalid_shape():
    """hmm_parameters_dict with mu=None should still validate other parameters."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    hmm_params = {
        'mu': None,
        'sigma': np.array([1.0, 1.0, 2.0]),
        'init': np.array([0.5, 0.5]),
        'trans': np.array([[0.95, 0.05], [0.05, 0.95]]),
    }

    with pytest.raises(ValueError, match='mu.*must have shape'):
        ihmm(
            velocities=velocities,
            hmm_parameters_dict=hmm_params,
            minimum_duration=1,
        )


def test_ihmm_rejects_transition_probabilities_row_sums_greater_than_one():
    """Transition probabilities with row sum > 1 should raise ValueError."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    transition_probabilities = np.array([[0.6, 0.6], [0.9, 0.2]])

    with pytest.raises(ValueError, match='transition_probabilities values must sum up to one'):
        ihmm(
            velocities=velocities,
            transition_probabilities=transition_probabilities,
            minimum_duration=1,
        )


def test_ihmm_rejects_sigma_with_invalid_shape():
    """Sigma parameter with invalid shape should raise ValueError."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    sigma = np.array([1.0, 1.0, 2.0])  # shape (3,) instead of (2,)

    with pytest.raises(ValueError, match='sigma.*must have shape'):
        ihmm(
            velocities=velocities,
            sigma=sigma,
            minimum_duration=1,
        )


def test_ihmm_rejects_init_state_with_invalid_shape():
    """init_state parameter with invalid shape should raise ValueError."""
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    init_state = np.array([0.5, 0.5, 0.0])  # shape (3,) instead of (2,)

    with pytest.raises(ValueError, match='init_state.*must have shape'):
        ihmm(
            velocities=velocities,
            init_state=init_state,
            minimum_duration=1,
        )


def test_ihmm_handles_polars_series_with_correct_structure():
    """Polars Series with 2D list structure should be processed correctly."""
    # Create a polars Series with 2D velocity data
    velocity_data = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
    series = pl.Series(velocity_data)

    events = ihmm(
        velocities=series,
        minimum_duration=1,
    )

    assert events is not None
    assert len(events.frame) >= 1


def test_ihmm_rejects_polars_series_with_non_list_dtype():
    """Polars Series with non-list dtype should raise TypeError."""
    series = pl.Series([1, 2, 3, 4])  # int dtype, not List

    with pytest.raises(TypeError, match='velocities dtype must be List'):
        ihmm(
            velocities=series,
            minimum_duration=1,
        )


def test_ihmm_rejects_polars_series_with_inconsistent_list_lengths():
    """Polars Series with inconsistent list lengths should raise ValueError."""
    # Some lists have length 2, others have length 3
    velocity_data = [[0.0, 0.0], [0.1, 0.1, 0.2], [10.0, 10.0]]
    series = pl.Series(velocity_data)

    with pytest.raises(ValueError, match='velocities must be 2D list'):
        ihmm(
            velocities=series,
            minimum_duration=1,
        )


def test_ihmm_warns_when_verbose_true_without_reestimation():
    """Verbose=True with reestimation=False should issue a warning."""
    velocities = np.array([[0.0, 0.0], [0.1, 0.1]])

    with pytest.warns(UserWarning, match='verbose is:True but reestimation is False'):
        ihmm(
            velocities=velocities,
            verbose=True,
            reestimation=False,
            minimum_duration=1,
        )


def test_ihmm_verbose_output_with_reestimation(capsys):
    """Verbose output should be printed when reestimation=True and verbose=True."""
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ],
    )

    ihmm(
        velocities=velocities,
        reestimation=True,
        reestimation_max_iters=5,
        verbose=True,
        minimum_duration=1,
    )

    captured = capsys.readouterr()
    assert 'Optimal parameters found by reestimation are:' in captured.out


def test_baum_welch_handles_masked_velocities_in_xi_computation():
    """Baum-Welch should handle masked velocities in the xi computation else branch."""
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([0.0, np.nan, 10.0])
    mask = np.array([True, False, True])

    result = _baum_welch(
        states=2,
        mu=mu.copy(),
        sigma=sigma.copy(),
        init=init.copy(),
        trans=trans.copy(),
        velocities=velocities,
        velocities_mask=mask,
        max_iters=5,
    )

    assert result['mu'].shape == (2,)
    assert result['sigma'].shape == (2,)
    assert all(np.isfinite(result['trans'].flatten()))


def test_baum_forward_initialization_with_masked_first_observation():
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.7, 0.3]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([np.nan, 0.1, 10.0])
    mask = np.array([False, True, True])  # First observation is masked

    alpha = _baum_forward(
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
        T=3,
        M=2,
    )

    expected_alpha_0 = init.copy()

    np.testing.assert_array_almost_equal(alpha[0], expected_alpha_0, decimal=12)

    assert np.all(np.isfinite(alpha[1]))
    assert np.all(np.isfinite(alpha[2]))


def test_baum_forward_initialization_with_unmasked_first_observation():

    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.7, 0.3]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([0.0, 0.1, 10.0])
    mask = np.array([True, True, True])  # All observations are unmasked

    alpha = _baum_forward(
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
        T=3,
        M=2,
    )

    expected_alpha_0 = np.array([
        init[0] + _emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=0),
        init[1] + _emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=1),
    ])

    np.testing.assert_array_almost_equal(alpha[0], expected_alpha_0, decimal=12)

    init_only = init.copy()
    assert not np.allclose(alpha[0], init_only), \
        'alpha[0] should include emission probability, not just init'


def test_baum_forward_initialization_masked_vs_unmasked_comparison():

    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([5.0, 0.1, 10.0])
    mask_unmasked = np.array([True, True, True])
    alpha_unmasked = _baum_forward(
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask_unmasked,
        T=3,
        M=2,
    )

    mask_masked = np.array([False, True, True])
    alpha_masked = _baum_forward(
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask_masked,
        T=3,
        M=2,
    )

    np.testing.assert_array_almost_equal(alpha_masked[0], init, decimal=12)

    assert not np.allclose(alpha_unmasked[0], alpha_masked[0])

    assert np.all(np.isfinite(alpha_unmasked[2]))
    assert np.all(np.isfinite(alpha_masked[2]))


def test_baum_forward_handles_mixed_masked_unmasked_observations():

    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([np.nan, 0.1, 10.0, 10.1])
    mask = np.array([False, True, True, True])  # Only first is masked

    alpha = _baum_forward(
        mu=mu,
        sigma=sigma,
        init=init,
        trans=trans,
        velocities=velocities,
        velocities_mask=mask,
        T=4,
        M=2,
    )

    assert np.allclose(alpha[0], init)

    assert np.all(np.isfinite(alpha[1]))
    assert np.all(np.isfinite(alpha[2]))
    assert np.all(np.isfinite(alpha[3]))

    assert not np.allclose(alpha[0], alpha[1])
    assert not np.allclose(alpha[1], alpha[2])
    assert not np.allclose(alpha[2], alpha[3])
