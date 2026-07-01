import numpy as np
import pytest

from pymovements.events.detection.ihmm import (
    emit_log_prob,
    log_sum_exp,
    baum_forward,
    baum_backward,
    baum_welch,
    viterbi,
    collapse_states,
    compute_hmm,
    ihmm,
    format_optimal_dict,
)


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
        1 + np.exp(-1.0) + np.exp(-2.0)
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


@pytest.fixture
def simple_hmm_params():
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])

    init = np.log(np.array([0.5, 0.5]))

    trans = np.log(
        np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ]
        )
    )

    velocities = np.array([0.1, -0.2, 9.9, 10.2])
    mask = np.array([True, True, True, True])

    return {
        "mu": mu,
        "sigma": sigma,
        "init": init,
        "trans": trans,
        "velocities": velocities,
        "mask": mask,
    }


def test_baum_forward_shape(simple_hmm_params):
    params = simple_hmm_params

    alpha = baum_forward(
        mu=params["mu"],
        sigma=params["sigma"],
        init=params["init"],
        trans=params["trans"],
        velocities=params["velocities"],
        velocities_mask=params["mask"],
        T=len(params["velocities"]),
        M=2,
    )

    assert alpha.shape == (4, 2)


def test_baum_backward_shape(simple_hmm_params):
    params = simple_hmm_params

    beta = baum_backward(
        mu=params["mu"],
        sigma=params["sigma"],
        trans=params["trans"],
        velocities=params["velocities"],
        velocities_mask=params["mask"],
        T=len(params["velocities"]),
        M=2,
    )

    assert beta.shape == (4, 2)


def test_forward_backward_produce_same_log_likelihood(simple_hmm_params):
    """Forward and backward algorithms must agree on sequence likelihood."""
    params = simple_hmm_params

    alpha = baum_forward(
        mu=params["mu"],
        sigma=params["sigma"],
        init=params["init"],
        trans=params["trans"],
        velocities=params["velocities"],
        velocities_mask=params["mask"],
        T=len(params["velocities"]),
        M=2,
    )

    beta = baum_backward(
        mu=params["mu"],
        sigma=params["sigma"],
        trans=params["trans"],
        velocities=params["velocities"],
        velocities_mask=params["mask"],
        T=len(params["velocities"]),
        M=2,
    )

    forward_ll = log_sum_exp(alpha[-1])

    backward_terms = []
    for s in range(2):
        backward_terms.append(
            params["init"][s]
            + emit_log_prob(
                mu=params["mu"],
                sigma=params["sigma"],
                v=params["velocities"][0],
                s=s,
            )
            + beta[0, s]
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
            ]
        )
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
            ]
        )
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
    
    timesteps = np.array([0, 10, 20, 30, 40, 50, 60, 70,80])

    
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
            ]
        )
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

    assert result["mu"].shape == (2,)
    assert result["sigma"].shape == (2,)
    assert result["init"].shape == (2,)
    assert result["trans"].shape == (2, 2)


def test_baum_welch_transition_rows_sum_to_one_in_probability_space():
    mu = np.array([0.0, 10.0])
    sigma = np.array([1.0, 1.0])

    init = np.log(np.array([0.5, 0.5]))

    trans = np.log(
        np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        )
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

    trans_probs = np.exp(result["trans"])

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
        ]
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

    estimated_mu = np.sort(result["mu"])

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
        ]
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
        ]
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
        ]
    )

    timesteps = np.array([0.0, 1.5])

    with pytest.raises(TypeError, match="timesteps must be of type int"):
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
        ]
    )

    with pytest.raises(ValueError, match="mu"):
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
        ]
    )

    with pytest.raises(ValueError, match="transition_probabilities"):
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
        ]
    )

    events = ihmm(
        velocities=velocities,
        minimum_duration=1,
    )

    assert events is not None


def test_ihmm_rejects_non_integer_minimum_duration():
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(TypeError, match="minimum_duration must be of type int"):
        ihmm(
            velocities=velocities,
            minimum_duration=1.5,
        )


def test_ihmm_rejects_zero_or_negative_minimum_duration():
    velocities = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="minimum_duration must be greater than 0"):
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
        ]
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

    with pytest.raises(ValueError, match="hmm_parameters_dict"):
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
        ]
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
        ]
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
            ]
        )
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