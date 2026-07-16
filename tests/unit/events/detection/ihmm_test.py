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
import numpy as np
import pandas as pd
import polars as pl
import pytest

import pymovements as pm
from pymovements.events.detection.idt import idt
from pymovements.events.detection.ihmm import baum_backward
from pymovements.events.detection.ihmm import baum_forward
from pymovements.events.detection.ihmm import baum_welch
from pymovements.events.detection.ihmm import collapse_states
from pymovements.events.detection.ihmm import compute_hmm
from pymovements.events.detection.ihmm import emit_log_prob
from pymovements.events.detection.ihmm import format_optimal_dict
from pymovements.events.detection.ihmm import ihmm
from pymovements.events.detection.ihmm import log_sum_exp
from pymovements.events.detection.ihmm import viterbi
from pymovements.synthetic import step_function
from pymovements.transforms.numpy import pos2vel


# -----------------------------------------------------------------------------
# emit_log_prob
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

    result = emit_log_prob(mu=mu, sigma=sigma, v=v, s=s)

    assert np.isclose(result, expected, atol=1e-12)


def test_emit_log_prob_uses_correct_state_parameters():
    """Different states should yield different probabilities."""
    mu = np.array([0.0, 100.0])
    sigma = np.array([1.0, 1.0])

    v = 0.0

    state0 = emit_log_prob(mu=mu, sigma=sigma, v=v, s=0)
    state1 = emit_log_prob(mu=mu, sigma=sigma, v=v, s=1)

    assert state0 > state1


def test_emit_log_prob_sigma_floor_prevents_instability():
    """Very small sigma should not produce NaN or inf."""
    mu = np.array([0.0, 0.0])
    sigma = np.array([0.0, 1.0])

    result = emit_log_prob(mu=mu, sigma=sigma, v=0.0, s=0)

    assert np.isfinite(result)


# -----------------------------------------------------------------------------
# log_sum_exp
# -----------------------------------------------------------------------------


def test_log_sum_exp_matches_manual_computation():
    arr = np.array([-2.0, -1.0, -0.5])

    expected = np.log(np.sum(np.exp(arr)))

    result = log_sum_exp(arr)

    assert np.isclose(result, expected, atol=1e-12)


def test_log_sum_exp_is_numerically_stable():
    """Large negative values should still produce finite output."""
    arr = np.array([-1000.0, -1001.0, -1002.0])

    result = log_sum_exp(arr)

    expected = -1000.0 + np.log(
        1 + np.exp(-1.0) + np.exp(-2.0),
    )

    assert np.isfinite(result)
    assert np.isclose(result, expected, atol=1e-10)


# -----------------------------------------------------------------------------
# format_optimal_dict
# -----------------------------------------------------------------------------


def test_format_optimal_dict_converts_to_json_serializable():
    """Should convert numpy arrays to lists of floats and exponentiate log probs."""
    opt = {
        'mu': np.array([1.0, 2.0]),
        'sigma': np.array([0.5, 0.5]),
        'init': np.log(np.array([0.7, 0.3])),
        'trans': np.log(np.array([[0.9, 0.1], [0.2, 0.8]])),
    }

    result = format_optimal_dict(opt)

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

    alpha = baum_forward(
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

    beta = baum_backward(
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

    alpha = baum_forward(
        mu=params['mu'],
        sigma=params['sigma'],
        init=params['init'],
        trans=params['trans'],
        velocities=params['velocities'],
        velocities_mask=params['mask'],
        T=len(params['velocities']),
        M=2,
    )

    beta = baum_backward(
        mu=params['mu'],
        sigma=params['sigma'],
        trans=params['trans'],
        velocities=params['velocities'],
        velocities_mask=params['mask'],
        T=len(params['velocities']),
        M=2,
    )

    forward_ll = log_sum_exp(alpha[-1])

    backward_terms = []
    for s in range(2):
        backward_terms.append(
            params['init'][s]
            + emit_log_prob(
                mu=params['mu'],
                sigma=params['sigma'],
                v=params['velocities'][0],
                s=s,
            )
            + beta[0, s],
        )

    backward_ll = log_sum_exp(np.array(backward_terms))

    assert np.isclose(forward_ll, backward_ll, atol=1e-10)


def test_forward_handles_masked_values():
    """Masked observations should skip emission contribution."""
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])
    init = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))

    velocities = np.array([0.0, np.nan, 10.0])
    mask = np.array([True, False, True])

    alpha = baum_forward(
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

    states = viterbi(
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

    states = viterbi(
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


# -----------------------------------------------------------------------------
# collapse_states
# -----------------------------------------------------------------------------


def test_collapse_states_extracts_fixation_segments():
    states = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    timesteps = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    onsets, offsets = collapse_states(states, timesteps)

    np.testing.assert_array_equal(onsets, np.array([1, 4]))
    np.testing.assert_array_equal(offsets, np.array([2, 6]))


def test_collapse_states_handles_full_fixation_sequence():
    states = np.array([0, 0, 0, 0])
    timesteps = np.array([0, 1, 2, 3])

    onsets, offsets = collapse_states(states, timesteps)

    np.testing.assert_array_equal(onsets, np.array([0]))
    np.testing.assert_array_equal(offsets, np.array([3]))


def test_collapse_states_handles_no_fixations():
    states = np.array([1, 1, 1, 1])
    timesteps = np.array([0, 1, 2, 3])

    onsets, offsets = collapse_states(states, timesteps)

    assert len(onsets) == 0
    assert len(offsets) == 0


def test_collapse_states_respects_min_duration():
    """Fixations shorter than min_duration should be filtered out."""

    states = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])

    timesteps = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])

    onsets, offsets = collapse_states(states, timesteps, min_duration=20)

    np.testing.assert_array_equal(onsets, np.array([40]))
    np.testing.assert_array_equal(offsets, np.array([60]))


def test_collapse_states_handles_empty_input():
    """Empty arrays should return empty arrays."""
    onsets, offsets = collapse_states(np.array([]), np.array([]))

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

    result = baum_welch(
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

    result = baum_welch(
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

    result = baum_welch(
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


# -----------------------------------------------------------------------------
# compute_hmm
# -----------------------------------------------------------------------------


def test_compute_hmm_returns_one_state_per_observation():
    velocities = np.array([0.0, 0.1, 10.0, 10.1])
    mask = np.array([True, True, True, True])

    states = compute_hmm(
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

    states = compute_hmm(
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

    states = compute_hmm(
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


def test_ihmm_rejects_fractional_timesteps():
    velocities = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
    )

    timesteps = np.array([0.0, 1.5])

    with pytest.raises(TypeError, match='timesteps must be of type int'):
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


def test_ihmm_handles_nan_velocities():
    velocities = np.array(
        [
            [0.0, 0.0],
            [np.nan, np.nan],
            [10.0, 10.0],
        ],
    )

    events = ihmm(
        velocities=velocities,
        minimum_duration=1,
    )

    assert events is not None


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


def test_ihmm_with_reestimation():
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

    result = emit_log_prob(
        mu=mu,
        sigma=sigma,
        v=0.0,
        s=0,
    )

    expected = -0.9189385332046727

    assert np.isclose(result, expected, atol=1e-12)


def test_log_sum_exp_regression_value():
    arr = np.log(np.array([1.0, 2.0, 3.0]))

    result = log_sum_exp(arr)

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

    result = viterbi(
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
# format_optimal_dict regression
# -----------------------------------------------------------------------------


def test_format_optimal_dict_regression():
    """Regression test for format_optimal_dict output structure."""
    opt = {
        'mu': np.array([2.0, 8.0]),
        'sigma': np.array([1.5, 2.5]),
        'init': np.log(np.array([0.6, 0.4])),
        'trans': np.log(np.array([[0.85, 0.15], [0.25, 0.75]])),
    }

    result = format_optimal_dict(opt)

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


test_results = []


def record(name, passed, detail=''):
    test_results.append({
        'test': name, 'passed': bool(
            passed,
        ) if passed is not None else None, 'detail': detail,
    })
    status = 'PASS' if passed else 'FAIL'
    if passed is None:
        status = 'SKIP'
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ''))


def to_frame(events):
    """Return the underlying table (Polars/Pandas-like) for an events result."""
    return events.frame if hasattr(events, 'frame') else events


def _column(frame, col):
    series = frame[col]
    return series.to_list() if hasattr(series, 'to_list') else list(series)


def n_events(events):
    return len(to_frame(events))


def spans(events):
    """List of (onset, offset) tuples for every detected event."""
    frame = to_frame(events)
    if len(frame) == 0:
        return []
    onsets = _column(frame, 'onset')
    offsets = _column(frame, 'offset')
    return list(zip(onsets, offsets))


def show(label, events):
    print(f"--- {label} ({n_events(events)} event(s)) ---")
    print(to_frame(events))


def overlap_ratio(span_a, span_b):
    """Intersection-over-union of two (onset, offset) intervals."""
    a0, a1 = span_a
    b0, b1 = span_b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def best_overlaps(spans_a, spans_b):
    """For each span in a, the best IoU achievable against any span in b."""
    if not spans_a:
        return []
    return [max((overlap_ratio(sa, sb) for sb in spans_b), default=0.0) for sa in spans_a]


def assert_event_count(name, events, expected):
    actual = n_events(events)
    passed = actual == expected
    record(f"{name}: event count == {expected}", passed, f"got {actual}")
    assert passed, f"{name}: expected {expected} event(s), got {actual}"


def assert_algorithms_agree(name, ihmm_events, idt_events, min_overlap=0.5):
    """Same number of fixations, and each ihmm fixation overlaps some idt fixation well."""
    ihmm_spans, idt_spans = spans(ihmm_events), spans(idt_events)
    count_match = len(ihmm_spans) == len(idt_spans)
    record(
        f"{name}: fixation counts match",
        count_match,
        f"ihmm={len(ihmm_spans)}, idt={len(idt_spans)}",
    )
    overlaps = best_overlaps(ihmm_spans, idt_spans)
    worst = min(overlaps) if overlaps else 1.0
    ok = worst >= min_overlap
    record(f"{name}: min pairwise overlap >= {min_overlap}", ok, f"worst overlap={worst:.2f}")
    return count_match and ok


def test_case1_no_fixations():
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

    ihmmEvents = ihmm(velocities_no_fix, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_no_fix, name='idt_fix')

    show('IHMM — no fixations', ihmmEvents)
    show('IDT — no fixations', idtEvents)

    assert_event_count('No fixations / ihmm', ihmmEvents, 0)
    assert_event_count('No fixations / idt', idtEvents, 0)


def test_case2_one_fixation():
    length_one_fix = 200
    positions_one_fix = step_function(
        length=length_one_fix,
        steps=[1],
        values=[(50., 50.)],
        start_value=(0., 0.),
    )

    velocities_one_fix = pos2vel(positions_one_fix, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_one_fix, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_one_fix, name='idt_fix')

    show('IHMM — one fixation', ihmmEvents)
    show('IDT — one fixation', idtEvents)

    assert_event_count('One fixation / ihmm', ihmmEvents, 1)
    assert_event_count('One fixation / idt', idtEvents, 1)
    assert_algorithms_agree('One fixation', ihmmEvents, idtEvents, min_overlap=0.9)


def test_case3_one_big_fixation():
    length_big_fix = 200
    positions_big_fix = np.tile(np.array([[75., 75.]]), (length_big_fix, 1))

    velocities_big_fix = pos2vel(positions_big_fix, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_big_fix, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_big_fix, name='idt_fix')

    show('IHMM — one big fixation', ihmmEvents)
    show('IDT — one big fixation', idtEvents)

    assert_event_count('One big fixation / ihmm', ihmmEvents, 1)
    assert_event_count('One big fixation / idt', idtEvents, 1)

    # The single fixation should cover (nearly) the whole recording.
    min_span = length_big_fix * 0.9
    for label, events in (('ihmm', ihmmEvents), ('idt', idtEvents)):
        onset, offset = spans(events)[0]
        covered = offset - onset
        ok = covered >= min_span
        record(f"One big fixation / {label} covers >= 90% of trial", ok, f"span={covered}")
        assert ok

    assert_algorithms_agree('One big fixation', ihmmEvents, idtEvents, min_overlap=0.95)


def test_case4_trailing_and_ending_fixations():
    length_edges = 200
    split_at = 90
    positions_edges = step_function(
        length=length_edges,
        steps=[split_at],
        values=[(60., 60.)],
        start_value=(10., 10.),
    )

    velocities_edges = pos2vel(positions_edges, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_edges, reestimation=True, name='ihmm_fix', minimum_duration=2)
    idtEvents = idt(positions_edges, name='idt_fix', minimum_duration=2)

    show('IHMM — trailing/ending fixations', ihmmEvents)
    show('IDT — trailing/ending fixations', idtEvents)

    assert_event_count('Trailing/ending fixations / ihmm', ihmmEvents, 2)
    assert_event_count('Trailing/ending fixations / idt', idtEvents, 2)

    for label, events in (('ihmm', ihmmEvents), ('idt', idtEvents)):
        event_spans = sorted(spans(events), key=lambda s: s[0])
        first_onset = event_spans[0][0]
        last_offset = event_spans[-1][1]
        starts_at_edge = first_onset <= 1  # allow 1 sample of slack (ms at 1000Hz)
        ends_at_edge = last_offset >= (length_edges - 1) * (1000.0 / sampling_rate) - 1
        record(
            f"Trailing/ending / {label}: first fixation starts at trial start",
            starts_at_edge,
            f"onset={first_onset}",
        )
        record(
            f"Trailing/ending / {label}: last fixation ends at trial end",
            ends_at_edge,
            f"offset={last_offset}",
        )
        assert starts_at_edge and ends_at_edge

    assert_algorithms_agree('Trailing/ending fixations', ihmmEvents, idtEvents, min_overlap=0.85)


def test_case5_toy_dataset():
    toy_available = True

    dataset = pm.Dataset('ToyDataset', path='data/ToyDataset')
    dataset.download()
    dataset.load()

    if toy_available:
        dataset.pix2deg()
        dataset.pos2vel()

        toy_gaze = dataset.gaze[0]
        toy_positions = toy_gaze.samples.select('position').to_series().to_list()
        toy_velocities = toy_gaze.samples.select('velocity').to_series().to_list()

        toy_velocities = [[np.nan, np.nan] if v == [None, None] else v for v in toy_velocities]
        print(toy_velocities)

        ihmmEvents = ihmm(toy_velocities, reestimation=True, name='ihmm_fix')
        idtEvents = idt(np.array(toy_positions), name='idt_fix')

        show('IHMM — toy dataset', ihmmEvents)
        show('IDT — toy dataset', idtEvents)

    if toy_available:
        ihmm_count, idt_count = n_events(ihmmEvents), n_events(idtEvents)
        larger, smaller = max(ihmm_count, idt_count), max(min(ihmm_count, idt_count), 1)
        relative_diff = (larger - smaller) / smaller
        ok = relative_diff <= 0.5  # allow up to 50% relative disagreement on real, noisy data
        record(
            'Toy dataset: fixation counts are comparable',
            ok,
            f"ihmm={ihmm_count}, idt={idt_count}, relative diff={relative_diff:.2f}",
        )
    else:
        record('Toy dataset: fixation counts are comparable', None, 'skipped (no network access)')


def test_case6_regular_dataset():
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

    ihmmEvents = ihmm(velocities_realistic, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_realistic, name='idt_fix')

    show('IHMM — regular dataset', ihmmEvents)
    show('IDT — regular dataset', idtEvents)

    expected_fixations = len(fixation_targets)

    for label, events in (('ihmm', ihmmEvents), ('idt', idtEvents)):
        actual = n_events(events)
        ok = abs(actual - expected_fixations) <= 1  # allow off-by-one at the boundaries
        record(
            f"Regular dataset / {label}: ~{expected_fixations} fixations detected",
            ok,
            f"got {actual}",
        )
        assert ok

    assert_algorithms_agree('Regular dataset', ihmmEvents, idtEvents, min_overlap=0.6)


#---

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
    """sigma parameter with invalid shape should raise ValueError."""
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
    """polars Series with 2D list structure should be processed correctly."""

    # Create a polars Series with 2D velocity data
    velocity_data = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
    series = pl.Series(velocity_data)

    events = ihmm(
        velocities=series,
        minimum_duration=1,
    )

    assert events is not None
    assert n_events(events) >= 1


def test_ihmm_rejects_polars_series_with_non_list_dtype():
    """polars Series with non-list dtype should raise TypeError."""

    series = pl.Series([1, 2, 3, 4])  # int dtype, not List

    with pytest.raises(TypeError, match='velocities dtype must be List'):
        ihmm(
            velocities=series,
            minimum_duration=1,
        )


def test_ihmm_rejects_polars_series_with_inconsistent_list_lengths():
    """polars Series with inconsistent list lengths should raise ValueError."""

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

    result = baum_welch(
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

    alpha = baum_forward(
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

    alpha = baum_forward(
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
        init[0] + emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=0),
        init[1] + emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=1),
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
    alpha_unmasked = baum_forward(
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
    alpha_masked = baum_forward(
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

    alpha = baum_forward(
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


@pytest.fixture(scope='session', autouse=True)
def _print_summary_at_session_end():
    yield
    summary = pd.DataFrame(test_results)
    print(summary)

    n_run = summary['passed'].notna().sum()
    n_passed = summary['passed'].sum()  # True counts as 1, False as 0
    n_skipped = summary['passed'].isna().sum()
    print(f"{n_passed}/{n_run} checks passed" + (f", {n_skipped} skipped" if n_skipped else ''))


# ---
