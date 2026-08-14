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
"""Provides the implementation for I-HMM algorithm."""
from __future__ import annotations

import warnings
from typing import Any

import numpy
import polars

from pymovements._utils import _checks
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events
from pymovements.transforms.numpy import norm


def _format_optimal_dict(opt: dict[str, Any]) -> dict[str, list[float] | list[list[float]]]:
    """Convert an optimization result dictionary into a JSON-serializable format.

    This function extracts model parameters from the input dictionary, converts
    NumPy scalar values into native Python floats, and exponentiates the
    logarithmic probability parameters (`init` and `trans`).

    Expected structure of `opt`:
        {
            "mu": array-like of shape (2,),
            "sigma": array-like of shape (2,),
            "init": array-like of shape (2,),          # log probabilities
            "trans": array-like of shape (2, 2),      # log transition probabilities
        }

    Parameters
    ----------
    opt: dict[str, Any]
        Dictionary containing optimization outputs. Values are expected
        to be NumPy arrays or array-like objects.

    Returns
    -------
    dict[str, list[float] | list[list[float]]]
        A dictionary with the following structure:
            {
                "mu": [float, float],
                "sigma": [float, float],
                "init": [float, float],
                "trans": [
                    [float, float],
                    [float, float],
                ],
            }

        The `init` and `trans` values are exponentiated before conversion.
    """
    out: dict[str, list[float] | list[list[float]]] = {}
    out['mu'] = [float(opt['mu'][0]), float(opt['mu'][1])]
    out['sigma'] = [float(opt['sigma'][0]), float(opt['sigma'][1])]
    out['init'] = [float(numpy.exp(opt['init'][0])), float(numpy.exp(opt['init'][1]))]
    out['trans'] = [
        [float(numpy.exp(opt['trans'][0][0])), float(numpy.exp(opt['trans'][0][1]))], [
            float(
                numpy.exp(opt['trans'][1][0]),
            ), float(numpy.exp(opt['trans'][1][1])),
        ],
    ]
    return out


def _emit_log_prob(
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    v: float,
    s: int,
) -> float:
    """Compute the log-probability of observing value `v` under a Gaussian emission model.

    This function evaluates the log-density of a univariate normal distribution
    parameterized by state-dependent mean (`mu`) and standard deviation (`sigma`),
    selecting parameters corresponding to state index `s`.

    A small numerical floor is applied to `sigma` to ensure stability.

    The computed quantity is:

        log p(v | s) = -0.5 * log(2πσ²) - (v - μ)² / (2σ²)

    Parameters
    ----------
    mu: numpy.ndarray | None
        Array of means for each hidden state. Shape: (num_states,).
        Must not be None.
    sigma: numpy.ndarray | None
        Array of standard deviations for each hidden state. Shape: (num_states,).
        Must not be None.
    v: float
        Observed scalar value.
    s: int
        Index of the hidden state used to select the corresponding (mu, sigma).

    Returns
    -------
    float
        The log-probability (float) of observing `v` given state `s`
        under a Gaussian emission model.

    Raises
    ------
    ValueError
        If `mu` or `sigma` is None.
    """
    if mu is None or sigma is None:
        raise ValueError('mu and sigma must not be None to compute an emission log-probability')

    mu_s = mu[s]
    sigma_s = max(sigma[s], 1e-6)

    return -0.5 * numpy.log(2 * numpy.pi * sigma_s**2) - ((v - mu_s)**2) / (2 * sigma_s**2)


def _log_sum_exp(
    arr: numpy.ndarray,
) -> float:
    """Compute log-sum-exp.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array of log-values.

    Returns
    -------
    float
        Logarithm of the summed exponentials.
    """
    m = numpy.max(arr)
    return m + numpy.log(numpy.sum(numpy.exp(arr - m)))


def _baum_welch(
    states: int,
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    init: numpy.ndarray | None,
    trans: numpy.ndarray | None,
    velocities: list[float] | numpy.ndarray,
    velocities_mask: numpy.ndarray,
    max_iters: int,
    epsilon: float = 1e-4,
) -> dict[str, numpy.ndarray]:
    """Estimate Hidden Markov Model parameters using the Baum-Welch algorithm.

    The Baum-Welch algorithm is an expectation-maximization (EM) algorithm used to
    find the maximum likelihood estimates of HMM parameters. This implementation
    handles partially observed velocity data through a masking mechanism.

    Parameters
    ----------
    states : int
        Number of hidden states in the HMM (M).

    mu : numpy.ndarray | None
        Initial means for the observation distributions (Gaussian emissions).
        Shape: (states,). Must not be None.

    sigma : numpy.ndarray | None
        Initial standard deviations for the observation distributions.
        Shape: (states,). Must not be None.

    init : numpy.ndarray | None
        Initial state probability distribution (log-space).
        Shape: (states,). Must not be None.

    trans : numpy.ndarray | None
        Initial state transition probability matrix (log-space).
        Shape: (states, states). trans[i, j] = log P(state_j | state_i). Must not be None.

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : numpy.ndarray
        Boolean mask indicating which velocity observations are valid/observed.
        Same length as velocities. True indicates observed, False indicates missing.

    max_iters : int
        Maximum number of EM iterations to perform.

    epsilon : float
        Convergence threshold. Algorithm stops when the absolute change in
        log-likelihood between iterations is less than this value. (default: 1e-4)

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary containing the estimated HMM parameters:

        - 'mu' : numpy.ndarray
            Estimated emission means for each state. Shape: (states,)
        - 'sigma' : numpy.ndarray
            Estimated emission standard deviations for each state. Shape: (states,)
        - 'init' : numpy.ndarray
            Estimated initial state probabilities (log-space). Shape: (states,)
        - 'trans' : numpy.ndarray
            Estimated state transition probabilities (log-space). Shape: (states, states)

    Raises
    ------
    ValueError
        If `mu`, `sigma`, `init`, or `trans` is None.
    """
    if mu is None or sigma is None or init is None or trans is None:
        raise ValueError(
            'mu, sigma, init and trans must not be None to run Baum-Welch reestimation',
        )

    # copy parameters so the caller's arrays are not modified in place
    mu = numpy.array(mu, dtype=float)
    sigma = numpy.array(sigma, dtype=float)
    init = numpy.array(init, dtype=float)
    trans = numpy.array(trans, dtype=float)

    T = len(velocities)
    M = states

    prev_log_likelihood = -numpy.inf

    # main EM loop

    for _ in range(max_iters):

        # forward pass

        alpha = _baum_forward(
            mu=mu,
            sigma=sigma,
            trans=trans,
            init=init,
            velocities=velocities,
            velocities_mask=velocities_mask,
            T=T,
            M=M,
        )

        # backward pass

        beta = _baum_backward(
            mu=mu,
            sigma=sigma,
            trans=trans,
            velocities=velocities,
            velocities_mask=velocities_mask,
            T=T,
            M=M,
        )

        # e-step

        xi = numpy.zeros((M, M, T - 1))

        for t in range(T - 1):
            denom_terms = []

            for i in range(M):
                for j in range(M):
                    if velocities_mask[t + 1]:
                        denom_terms.append(
                            alpha[t, i] +
                            trans[i, j] +
                            _emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                            beta[t + 1, j],
                        )
                    else:
                        denom_terms.append(
                            alpha[t, i] +
                            trans[i, j] +
                            0.0 +
                            beta[t + 1, j],
                        )

            denom = _log_sum_exp(numpy.array(denom_terms))

            for i in range(M):
                for j in range(M):
                    if velocities_mask[t + 1]:
                        num = (
                            alpha[t, i] +
                            trans[i, j] +
                            _emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                            beta[t + 1, j]
                        )
                    else:
                        num = (
                            alpha[t, i] +
                            trans[i, j] +
                            0.0 +
                            beta[t + 1, j]
                        )

                    xi[i, j, t] = numpy.exp(num - denom)

        gamma = numpy.sum(xi, axis=1)

        gamma_full = numpy.zeros((M, T))
        gamma_full[:, :-1] = gamma

        last = alpha[T - 1] + beta[T - 1]
        last = numpy.exp(last - _log_sum_exp(last))
        gamma_full[:, -1] = last

        # m-step

        init = numpy.log(numpy.clip(gamma_full[:, 0], 1e-12, 1.0))

        # laplace smoothing for division by 0 errors
        eps = 1e-12
        for i in range(M):
            denom = numpy.sum(gamma_full[i, :-1])
            for j in range(M):
                numerator = numpy.sum(xi[i, j, :])
                trans[i, j] = numpy.log((numerator + eps) / (denom + eps * M))

        for j in range(M):

            mask = velocities_mask

            weights = gamma_full[j, mask]
            vals = numpy.asarray(velocities)[mask]

            total = numpy.sum(weights)

            # keep the previous emission parameters if a state carries no posterior
            # mass, otherwise the update would divide by zero and yield nan.
            if total > 0:
                mu[j] = numpy.sum(weights * vals) / total

                var = numpy.sum(weights * (vals - mu[j])**2) / total
                sigma[j] = numpy.sqrt(var)

        # compute log-likelihood for convergence check

        alpha_updated = _baum_forward(
            mu=mu,
            sigma=sigma,
            trans=trans,
            init=init,
            velocities=velocities,
            velocities_mask=velocities_mask,
            T=T,
            M=M,
        )

        log_likelihood = _log_sum_exp(alpha_updated[-1])

        if abs(log_likelihood - prev_log_likelihood) < epsilon:
            break

        prev_log_likelihood = log_likelihood

    return {'mu': mu, 'sigma': sigma, 'init': init, 'trans': trans}


def _baum_forward(
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    init: numpy.ndarray | None,
    trans: numpy.ndarray | None,
    velocities: list[float] | numpy.ndarray,
    velocities_mask: numpy.ndarray,
    T: int,
    M: int,
) -> numpy.ndarray:
    """Compute forward probabilities (alpha) for a Hidden Markov Model.

    The forward algorithm computes the probability of being in each hidden state
    at each time step given the observed sequence up to that point. This implementation
    handles partially observed data through a masking mechanism and uses log-space
    computations for numerical stability.

    Parameters
    ----------
    mu : numpy.ndarray | None
        Means of the emission distributions (Gaussian) for each state.
        Shape: (M,). Must not be None.

    sigma : numpy.ndarray | None
        Standard deviations of the emission distributions for each state.
        Shape: (M,). Must not be None.

    init : numpy.ndarray | None
        Initial state probability distribution (log-space).
        Shape: (M,). init[s] = log(P(state = s at time 0)). Must not be None.

    trans : numpy.ndarray | None
        State transition probability matrix (log-space).
        Shape: (M, M). trans[i, j] = log(P(state = j at time t | state = i at time t-1)).
        Must not be None.

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : numpy.ndarray
        Boolean mask indicating which velocity observations are valid/observed.
        Length: T. True indicates observed, False indicates missing.

    T : int
        Number of time steps (length of observation sequence).

    M : int
        Number of hidden states.

    Returns
    -------
    numpy.ndarray
        Forward probabilities (log-space). Shape: (T, M).
        alpha[t, s] = log(P(observations[0:t+1], state = s at time t | model parameters)).

    Raises
    ------
    ValueError
        If `mu`, `sigma`, `init`, or `trans` is None.
    """
    if mu is None or sigma is None or init is None or trans is None:
        raise ValueError(
            'mu, sigma, init and trans must not be None to compute forward probabilities',
        )

    alpha = numpy.full((T, M), -numpy.inf)

    # init step

    for s in range(M):
        if velocities_mask[0]:
            alpha[0, s] = init[s] + _emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=s)
        else:
            alpha[0, s] = init[s] + 0

    # induction step

    for t in range(1, T):
        for j in range(M):
            terms = []
            for i in range(M):
                terms.append(alpha[t - 1, i] + trans[i, j])
            if velocities_mask[t]:
                alpha[t, j] = _log_sum_exp(numpy.array(terms)) + \
                    _emit_log_prob(mu=mu, sigma=sigma, v=velocities[t], s=j)
            else:
                alpha[t, j] = _log_sum_exp(numpy.array(terms)) + \
                    0.0

    return alpha


def _baum_backward(
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    trans: numpy.ndarray | None,
    velocities: list[float] | numpy.ndarray,
    velocities_mask: numpy.ndarray,
    T: int,
    M: int,
) -> numpy.ndarray:
    """Compute backward probabilities (beta) for a Hidden Markov Model.

    The backward algorithm computes the probability of the future observation sequence
    given that the system is in a particular state at a particular time. This implementation
    handles partially observed data through a masking mechanism and uses log-space
    computations for numerical stability.

    Parameters
    ----------
    mu : numpy.ndarray | None
        Means of the emission distributions (Gaussian) for each state.
        Shape: (M,). Must not be None.

    sigma : numpy.ndarray | None
        Standard deviations of the emission distributions for each state.
        Shape: (M,). Must not be None.

    trans : numpy.ndarray | None
        State transition probability matrix (log-space).
        Shape: (M, M). trans[i, j] = log(P(state = j at time t+1 | state = i at time t)).
        Must not be None.

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : numpy.ndarray
        Boolean mask indicating which velocity observations are valid/observed.
        Length: T. True indicates observed, False indicates missing.

    T : int
        Number of time steps (length of observation sequence).

    M : int
        Number of hidden states.

    Returns
    -------
    numpy.ndarray
        Backward probabilities (log-space). Shape: (T, M).
        beta[t, i] = log(P(observations[t+1:T] | state = i at time t, model parameters)).

    Raises
    ------
    ValueError
        If `mu`, `sigma`, or `trans` is None.
    """
    if mu is None or sigma is None or trans is None:
        raise ValueError('mu, sigma and trans must not be None to compute backward probabilities')

    beta = numpy.full((T, M), -numpy.inf)

    # init step

    beta[T - 1, :] = 0

    # induction step

    for t in range(T - 2, -1, -1):
        for i in range(M):
            terms = []
            for j in range(M):
                if velocities_mask[t + 1]:
                    terms.append(
                        trans[i, j] +
                        _emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                        beta[t + 1, j],
                    )
                else:
                    terms.append(
                        trans[i, j] +
                        0.0 +
                        beta[t + 1, j],
                    )

            beta[t, i] = _log_sum_exp(numpy.array(terms))

    return beta


def _viterbi(
    states: int,
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    init: numpy.ndarray | None,
    trans: numpy.ndarray | None,
    velocities: list[float] | numpy.ndarray,
    velocities_mask: numpy.ndarray,
) -> numpy.ndarray:
    """Find the most likely sequence of hidden states using the Viterbi algorithm.

    The Viterbi algorithm is a dynamic programming algorithm that finds the
    most probable sequence of hidden states (the Viterbi path) given a sequence
    of observations. It uses the principle of optimality to efficiently compute
    the maximum probability path through the HMM lattice.

    Parameters
    ----------
    states : int
        Number of hidden states in the HMM (M).

    mu : numpy.ndarray | None
        Means of the emission distributions (Gaussian) for each state.
        Shape: (states,). Must not be None.

    sigma : numpy.ndarray | None
        Standard deviations of the emission distributions for each state.
        Shape: (states,). Must not be None.

    init : numpy.ndarray | None
        Initial state probability distribution (log-space).
        Shape: (states,). init[s] = log(P(state = s at time 0)). Must not be None.

    trans : numpy.ndarray | None
        State transition probability matrix (log-space).
        Shape: (states, states). trans[i, j] = log(P(state = j at time t | state = i at time t-1)).
        Must not be None.

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : numpy.ndarray
        Boolean mask indicating which velocity observations are valid/observed.
        Length: T. True indicates observed, False indicates missing.

    Returns
    -------
    numpy.ndarray
        Most likely sequence of hidden states (Viterbi path).
        Shape: (T,), dtype=int. Each entry is a state index from 0 to states-1.

    Raises
    ------
    ValueError
        If `mu`, `sigma`, `init`, or `trans` is None.
    """
    if mu is None or sigma is None or init is None or trans is None:
        raise ValueError('mu, sigma, init and trans must not be None to run Viterbi decoding')

    # init step

    T = len(velocities)

    prob = numpy.full((T, states), -numpy.inf)
    prev = numpy.zeros((T, states), dtype=int)

    for s in range(states):
        if velocities_mask[0]:
            prob[0, s] = init[s] + _emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=s)
        else:
            prob[0, s] = init[s]

    # main loop

    for t in range(1, T):
        for state1 in range(states):
            best_prob = -numpy.inf
            best_state = 0
            for state2 in range(states):
                if velocities_mask[t]:
                    new_prob = prob[t - 1, state2] + trans[state2, state1] + \
                        _emit_log_prob(mu=mu, sigma=sigma, v=velocities[t], s=state1)
                else:

                    new_prob = prob[t - 1, state2] + trans[state2, state1] + 0
                if new_prob > best_prob:
                    best_prob = new_prob
                    best_state = state2
            prob[t, state1] = best_prob
            prev[t, state1] = best_state

    # backtrack

    path = numpy.zeros(T, dtype=int)

    path[T - 1] = numpy.argmax(prob[T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = prev[t + 1, path[t + 1]]

    return path


def _collapse_states(
        states: numpy.ndarray,
        timesteps: numpy.ndarray,
        fixation_state: int = 0,
        min_duration: int = 0,

) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Extract contiguous fixation periods from a sequence of state labels.

    This function identifies consecutive runs of a specified fixation state and
    returns the onset and offset times for each fixation period. It collapses
    the detailed per-timestep state sequence into a list of fixation events.

    Parameters
    ----------
    states : numpy.ndarray
        Array of state labels for each timestep. Typically output from Viterbi
        or other HMM decoding methods. Shape: (T,), where T is number of timesteps.

    timesteps : numpy.ndarray
        Array of time values corresponding to each state label.
        Must have the same length as states. Shape: (T,).

    fixation_state : int
        The state label that represents fixation periods.
        All other states are ignored. (default: 0, commonly used for fixation).

    min_duration: int
        Minimum fixation duration. The duration should be the same unit as the timesteps array.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing two arrays:

        - onsets : numpy.ndarray
            Start times of each fixation period. Shape: (N,), where N is number
            of fixation periods.

        - offsets : numpy.ndarray
            End times of each fixation period. Shape: (N,).
            Same length as onsets.
    """
    if len(states) == 0 or len(timesteps) == 0:
        return numpy.array([]), numpy.array([])

    onsets = []
    offsets = []

    # iterate through the state sequence

    i = 0
    while i < len(states):

        # check if state is a fixation

        if states[i] == fixation_state:
            onset_idx = i
            onset_time = timesteps[onset_idx]

            # move forward to find the end of the fixation block

            j = i
            while j < len(states) and states[j] == fixation_state:
                j += 1

            offset_time = timesteps[j - 1]

            duration = offset_time - onset_time

            # only keep fixations that meet the minimum duration threshold

            if duration >= min_duration:
                onsets.append(onset_time)
                offsets.append(offset_time)

            i = j
        else:

            # current state is not a fixation so skip it

            i += 1

    return numpy.array(onsets), numpy.array(offsets)


def _compute_hmm(
    velocities: numpy.ndarray,
    verbose: bool,
    reestimation: bool,
    reestimation_max_iters: int,
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    init_state: numpy.ndarray | None,
    transition_probabilities: numpy.ndarray | None,
    velocities_mask: numpy.ndarray,
    hmm_parameters_dict: dict | None = None,
) -> numpy.ndarray:
    """Compute HMM state sequence for velocity data using optional parameter reestimation.

    This function serves as a high-level wrapper for HMM-based state decoding of
    velocity time series data. It handles parameter initialization, optional
    Baum-Welch reestimation, and Viterbi decoding to produce a sequence of hidden
    states (typically saccade vs. fixation).

    Parameters
    ----------
    velocities : numpy.ndarray
        Array of velocity measurements. Shape: (T,), where T is number of timesteps.

    verbose : bool
        If True, prints parameter values and reestimation results to console.

    reestimation : bool
        If True, performs Baum-Welch reestimation to optimize HMM parameters
        before state decoding.

    reestimation_max_iters : int
        Maximum number of EM iterations for Baum-Welch reestimation.
        Only used if reestimation is True.

    mu : numpy.ndarray | None
        Mean velocity for each state (Gaussian emissions).
        Shape: (2,), typically [fixation_mean, saccade_mean].
        If None, uses default or hmm_parameters_dict values.

    sigma : numpy.ndarray | None
        Standard deviation of velocity for each state.
        Shape: (2,), typically [fixation_std, saccade_std].
        If None, uses default or hmm_parameters_dict values.

    init_state : numpy.ndarray | None
        Initial state probability distribution (linear scale, not log).
        Shape: (2,), e.g., [0.5, 0.5].
        If None, uses default or hmm_parameters_dict values.

    transition_probabilities : numpy.ndarray | None
        State transition probability matrix (linear scale, not log).
        Shape: (2, 2), where trans[i, j] = P(state=j | state=i).
        If None, uses default or hmm_parameters_dict values.

    velocities_mask : numpy.ndarray
        Boolean mask indicating valid/observed velocity values.
        Shape: (T,). True for observed, False for missing/NaN values.

    hmm_parameters_dict : dict | None
        Dictionary containing custom HMM parameters with keys:
        - 'mu': list of 2 means
        - 'sigma': list of 2 standard deviations
        - 'init': list of 2 initial probabilities
        - 'trans': 2x2 transition probability matrix
        If None, uses data-driven defaults based on velocity percentiles.

    Returns
    -------
    numpy.ndarray
        Decoded state sequence. Shape: (T,), dtype=int.
        State 0 typically represents fixation, State 1 represents saccade.
    """
    # ignore nan values for default data driven initialization
    velocities_for_init = velocities[velocities_mask]

    # get or init parameters

    if hmm_parameters_dict is not None:
        defaults = hmm_parameters_dict
    else:
        # data driven initialization
        defaults = {
            'mu': [
                numpy.percentile(velocities_for_init, 30),
                numpy.percentile(velocities_for_init, 80),
            ],
            'sigma': [
                numpy.sqrt(numpy.var(velocities_for_init) / 2),
                numpy.sqrt(numpy.var(velocities_for_init)),
            ],
            'init': [0.5, 0.5],  # dummy average values should be fine for long sequences
            'trans': [[0.95, 0.05], [0.05, 0.95]],  # based on Salvucci's paper diagram
        }

    if mu is not None:
        _mu = mu
    else:
        _mu = defaults['mu']
    if sigma is not None:
        _sigma = sigma
    else:
        _sigma = defaults['sigma']
    if init_state is not None:
        _init = init_state
    else:
        _init = defaults['init']
    if transition_probabilities is not None:
        _trans = transition_probabilities
    else:
        _trans = defaults['trans']

    # clip before taking the log so that valid zero probabilities (e.g. an
    # init_state of [1, 0]) map to a large negative log value instead of
    # emitting a divide-by-zero warning from numpy.log.
    _init = numpy.log(numpy.clip(numpy.asarray(_init, dtype=float), 1e-12, 1.0))
    _trans = numpy.log(numpy.clip(numpy.asarray(_trans, dtype=float), 1e-12, 1.0))

    # reestimate if needed

    if reestimation:
        optimal = _baum_welch(
            states=2,
            mu=_mu,
            sigma=_sigma,
            init=_init,
            trans=_trans,
            velocities=velocities,
            velocities_mask=velocities_mask,
            max_iters=reestimation_max_iters,
        )

        _mu = optimal['mu']
        _sigma = optimal['sigma']
        _init = optimal['init']
        _trans = optimal['trans']

    # Enforce state order (state 0 = fixation = lowest mean velocity) so that the
    # decoded states carry a consistent meaning whether or not parameters were
    # re-estimated. Without this, user-supplied parameters given in saccade-first
    # order would silently mislabel fixations and saccades.

    _mu = numpy.asarray(_mu)
    _sigma = numpy.asarray(_sigma)
    order = numpy.argsort(_mu)

    _mu = _mu[order]
    _sigma = _sigma[order]
    _init = _init[order]
    _trans = _trans[order][:, order]

    if reestimation and verbose:
        optimal = {'mu': _mu, 'sigma': _sigma, 'init': _init, 'trans': _trans}
        print(f"Optimal parameters found by reestimation are:\n{_format_optimal_dict(optimal)}")

    # inference the hmm

    states = _viterbi(
        states=2,
        mu=_mu,
        sigma=_sigma,
        init=_init,
        trans=_trans,
        velocities=velocities,
        velocities_mask=velocities_mask,
    )

    return states


def _validate_transition_probabilities(transition_probabilities: numpy.ndarray) -> None:
    """Validate that a transition matrix is a valid row-stochastic probability matrix.

    Parameters
    ----------
    transition_probabilities: numpy.ndarray
        Transition probability matrix of shape ``(2, 2)``.

    Raises
    ------
    ValueError
        If any row does not sum to one, or if any entry lies outside ``[0, 1]``.
    """
    row_sums = numpy.sum(transition_probabilities, axis=1)
    if not numpy.allclose(row_sums, 1.0):
        raise ValueError(
            f'transition_probabilities values must sum up to one for each state '
            f'but instead are {row_sums[0]} and {row_sums[1]}',
        )
    if numpy.any(transition_probabilities < 0) or numpy.any(transition_probabilities > 1):
        raise ValueError(
            'transition_probabilities values must each lie between zero and one',
        )


def _validate_init_state(init_state: numpy.ndarray) -> None:
    """Validate that an initial-state vector is a valid probability distribution.

    Parameters
    ----------
    init_state: numpy.ndarray
        Initial state probability vector of shape ``(2,)``.

    Raises
    ------
    ValueError
        If the values do not sum to one, or if any entry lies outside ``[0, 1]``.
    """
    if not numpy.isclose(numpy.sum(init_state), 1.0):
        raise ValueError(
            f'init_state values must sum up to one '
            f'but instead sum up to {numpy.sum(init_state)}',
        )
    if numpy.any(init_state < 0) or numpy.any(init_state > 1):
        raise ValueError(
            'init_state values must each lie between zero and one',
        )


def _validate_hmm_parameters_dict(hmm_parameters_dict: dict) -> dict[str, numpy.ndarray]:
    """Validate and copy a user-supplied ``hmm_parameters_dict``.

    Checks that the dictionary has exactly the expected keys and that each value has
    the correct shape and, for ``init`` and ``trans``, forms a valid probability
    distribution. The values are copied into a new dictionary of numpy arrays so the
    caller's input is left untouched.

    Parameters
    ----------
    hmm_parameters_dict: dict
        Dictionary with keys ``'mu'``, ``'sigma'``, ``'init'`` and ``'trans'``.

    Returns
    -------
    dict[str, numpy.ndarray]
        A new dictionary with the same keys whose values are numpy arrays.

    Raises
    ------
    ValueError
        If the keys are incorrect, a parameter has the wrong shape, or the init/trans
        values are not valid probabilities.
    """
    if set(hmm_parameters_dict.keys()) != {'mu', 'sigma', 'init', 'trans'}:
        raise ValueError(
            f'hmm_parameters_dict'
            f' should have fields {["mu", "sigma", "init", "trans"]} but instead has '
            f'{list(hmm_parameters_dict.keys())}',
        )

    # copy into a new dict so the caller's input is not mutated
    validated = {
        'mu': numpy.array(hmm_parameters_dict['mu']),
        'sigma': numpy.array(hmm_parameters_dict['sigma']),
        'init': numpy.array(hmm_parameters_dict['init']),
        'trans': numpy.array(hmm_parameters_dict['trans']),
    }

    if validated['mu'].shape != (2,):
        raise ValueError(f'mu must have shape (2,), but shapes are {validated["mu"].shape}')
    if validated['sigma'].shape != (2,):
        raise ValueError(f'sigma must have shape (2,), but shapes are {validated["sigma"].shape}')
    if validated['init'].shape != (2,):
        raise ValueError(
            f'init_state must have shape (2,), but shapes are {validated["init"].shape}',
        )
    if validated['trans'].shape != (2, 2):
        raise ValueError(
            f'transition_probabilities must have shape (2, 2), but shapes are '
            f'{validated["trans"].shape}',
        )

    _validate_transition_probabilities(validated['trans'])
    _validate_init_state(validated['init'])

    return validated


@register_event_detection
def ihmm(
        velocities: list[list[float]] | list[tuple[float, float]] | numpy.ndarray | polars.Series,
        timesteps: list[int] | numpy.ndarray | polars.Series | None = None,
        minimum_duration: int = 100,
        mu: list[float] | numpy.ndarray | None = None,
        sigma: list[float] | numpy.ndarray | None = None,
        init_state: list[float] | numpy.ndarray | None = None,
        transition_probabilities: list[list[float]] | numpy.ndarray | None = None,
        reestimation_max_iters: int = 1000,
        reestimation: bool = False,
        verbose: bool = False,
        hmm_parameters_dict: dict | None = None,
        name: str = 'fixation',
) -> Events:
    """Detect fixation events from velocity data using a Hidden Markov Model (I-HMM).

    The implementation follows the algorithm from Salvucci and Goldberg
    :cite:p:`SalvucciGoldberg2000`.

    This function implements a 2-state HMM specifically designed for eye-tracking
    data to distinguish between fixations (state 0) and saccades (state 1). It
    processes velocity time series, estimates optimal parameters via Baum-Welch
    (optional), decodes the most likely state sequence using Viterbi, and collapses
    contiguous fixation periods into events.

    Parameters
    ----------
    velocities : list[list[float]] | list[tuple[float, float]] | numpy.ndarray | polars.Series
        Velocity data. Can be:
        - 2D array of shape (T, 2) containing x and y velocity components
        - List of (vx, vy) tuples or lists
        - polars Series of 2-element lists
        Must have shape (T, 2). Will be converted to velocity magnitudes via Euclidean norm.

    timesteps : list[int] | numpy.ndarray | polars.Series | None
        Timestamp for each velocity sample. May be integer or float valued.
        If None, uses sequential indices (0, 1, 2, ..., T-1). (default: None)

    minimum_duration: int
        Minimum fixation duration. The duration should be the same unit as the timesteps array.
        Must be an integer, so with float-valued ``timesteps`` (e.g. seconds) only
        whole-unit thresholds can be expressed. (default: 100)

    mu : list[float] | numpy.ndarray | None
        Mean velocity for each state (Gaussian emissions).
        Shape: (2,), typically [fixation_mean, saccade_mean].
        The state order is normalized internally by ascending mean, so the
        lowest-mean state is always treated as the fixation state (state 0)
        regardless of the order in which the two values are supplied.
        If None, uses data-driven defaults or hmm_parameters_dict. (default: None)

    sigma : list[float] | numpy.ndarray | None
        Standard deviation of velocity for each state.
        Shape: (2,), typically [fixation_std, saccade_std].
        If None, uses data-driven defaults or hmm_parameters_dict. (default: None)

    init_state : list[float] | numpy.ndarray | None
        Initial state probability distribution (linear scale).
        Shape: (2,), e.g., [0.5, 0.5]. Must sum to 1.
        If None, uses defaults or hmm_parameters_dict. (default: None)

    transition_probabilities : list[list[float]] | numpy.ndarray | None
        State transition probability matrix (linear scale).
        Shape: (2, 2). Each row must sum to 1.
        If None, uses default matrix [[0.95, 0.05], [0.05, 0.95]]. (default: None)

    reestimation_max_iters : int
        Maximum number of Baum-Welch EM iterations if reestimation=True. (default: 1000)

    reestimation : bool
        If True, performs Baum-Welch reestimation to optimize HMM parameters
        before state decoding. Recommended for robust parameter estimation. (default: False)

    verbose : bool
        If True, prints parameter values and reestimation progress.
        Only effective when reestimation=True. (default: False)

    hmm_parameters_dict : dict | None
        Dictionary containing custom HMM parameters with keys:
        - 'mu': list of 2 means
        - 'sigma': list of 2 standard deviations
        - 'init': list of 2 initial probabilities
        - 'trans': 2x2 transition probability matrix
        Overridden by explicit mu, sigma, init_state, transition_probabilities. (default: None)

    name : str
        Name for the detected events. Appears in the returned Events object.
        (default: 'fixation')

    Returns
    -------
    Events
        An Events object containing:
        - name: Event type name ('fixation' by default)
        - onsets: Array of fixation onset times
        - offsets: Array of fixation offset times
        Shape: (N,) where N is number of detected fixation events.

    Notes
    -----
    The processing pipeline consists of several steps:

    1. Input validation and conversion:
       - Converts velocities to 1D magnitude array via Euclidean norm
       - Removes leading/trailing NaN values
       - Validates parameter shapes and transition probability sums

    2. HMM parameter initialization (priority order):
       - Explicit parameters (mu, sigma, init_state, transition_probabilities)
       - Custom dictionary (hmm_parameters_dict)
       - Data-driven defaults (based on velocity percentiles)

    3. Optional parameter reestimation using Baum-Welch:
       - Maximizes likelihood of observed velocity data
       - Updates all HMM parameters
       - Runs for up to reestimation_max_iters iterations

    4. State decoding using Viterbi algorithm:
       - Finds most likely fixation/saccade sequence

    5. Event extraction:
       - Collapses consecutive fixation state periods into events
       - Returns onset and offset times for each fixation

    The default transition probabilities (0.95 for self-transitions, 0.05 for
    switches) are based on Salvucci's eye movement model, reflecting typical
    fixation and saccade durations.

    Only leading and trailing samples with missing (NaN) velocity are trimmed
    before decoding. Interior missing samples are kept, and their emission
    probability is skipped, so Viterbi assigns them a state based on the
    transition probabilities alone. As a result a fixation can span short gaps of
    missing data, and events are not split on interior NaNs. This differs from the
    ``include_nan`` option of :py:func:`~pymovements.events.detection.ivt`, which
    can split events on missing values.

    Raises
    ------
    TypeError
        If velocities is a polars Series whose dtype is not List.
        If timesteps is a polars Series with a non-numeric dtype.
        If minimum_duration is not an integer.
    ValueError
        If velocities does not have shape (T, 2).
        If velocities is a polars Series whose lists don't all have length 2.
        If parameter shapes are incorrect (not (2,) or (2,2)).
        If minimum_duration is not greater than 0.
        If transition_probabilities rows don't sum to 1, or contain values outside [0, 1].
        If init_state does not sum to 1, or contains values outside [0, 1].
        If hmm_parameters_dict has incorrect keys or shapes, or its init/trans
        values don't sum to 1 or lie outside [0, 1].

    Examples
    --------
    Create a synthetic step signal representing gaze segments.

    >>> import numpy as np
    >>> from pymovements.transforms.numpy import pos2vel
    >>> from pymovements.synthetic import step_function
    >>> from pymovements.gaze import from_numpy
    >>> positions = step_function(
    ...         length=500,
    ...         steps=[50, 250, 450, 650, 750],
    ...         values=[(5., 5.), (10., 10.), (5., 5.), (15., 15.), (5., 5.)],
    ...         start_value=(0., 0.))

    >>> positions.shape
    (500, 2)

    Transform into velocities

    >>> velocities = pos2vel(positions)
    >>> velocities.shape
    (500, 2)

    Apply event detection algorithm on numpy array:

    >>> ihmm(velocities)
    shape: (2, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 52    ┆ 247    ┆ 195      │
    │ fixation ┆ 252   ┆ 447    ┆ 195      │
    └──────────┴───────┴────────┴──────────┘

    Run fixation detection with custom HMM parameters:

    >>> hmm_parameters = {'mu': [2.0140785987072225, 69.41529375180251],
    ...         'sigma': [1.3220152347857494, 87.32409626093246],
    ...         'init': [1.e+00, 1.e-12],
    ...         'trans': [[0.97360507, 0.02639493],[0.07593547, 0.92406453]]}
    >>> ihmm(velocities, hmm_parameters_dict = hmm_parameters)
    shape: (2, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 52    ┆ 247    ┆ 195      │
    │ fixation ┆ 252   ┆ 447    ┆ 195      │
    └──────────┴───────┴────────┴──────────┘

    We can also apply the detection on a :py:class:`~pymovements.Gaze` object.

    >>> from pymovements import Experiment
    >>> gaze = from_numpy(
    ...         velocity=velocities.T,
    ...         time=np.arange(len(velocities)),)
    >>> gaze
    shape: (500, 2)
    ┌──────┬────────────┐
    │ time ┆ velocity   │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.0, 0.0] │
    │ 1    ┆ [0.0, 0.0] │
    │ 2    ┆ [0.0, 0.0] │
    │ 3    ┆ [0.0, 0.0] │
    │ 4    ┆ [0.0, 0.0] │
    │ …    ┆ …          │
    │ 495  ┆ [0.0, 0.0] │
    │ 496  ┆ [0.0, 0.0] │
    │ 497  ┆ [0.0, 0.0] │
    │ 498  ┆ [0.0, 0.0] │
    │ 499  ┆ [0.0, 0.0] │
    └──────┴────────────┘

    Run fixation detection by using the :py:meth:`~pymovements.Gaze.detect` method.

    >>> gaze.detect('ihmm')
    >>> gaze.events
    shape: (2, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 52    ┆ 247    ┆ 195      │
    │ fixation ┆ 252   ┆ 447    ┆ 195      │
    └──────────┴───────┴────────┴──────────┘

    Passing parameters to :py:meth:`~pymovements.Gaze.detect`:

    >>> gaze.detect('ihmm', reestimation=True, name='fixation_ihmm')
    >>> gaze.events.filter_by_name('fixation_ihmm')
    shape: (2, 4)
    ┌───────────────┬───────┬────────┬──────────┐
    │ name          ┆ onset ┆ offset ┆ duration │
    │ ---           ┆ ---   ┆ ---    ┆ ---      │
    │ str           ┆ i64   ┆ i64    ┆ i64      │
    ╞═══════════════╪═══════╪════════╪══════════╡
    │ fixation_ihmm ┆ 52    ┆ 247    ┆ 195      │
    │ fixation_ihmm ┆ 252   ┆ 447    ┆ 195      │
    └───────────────┴───────┴────────┴──────────┘
    """
    if isinstance(velocities, polars.Series):
        if not isinstance(velocities.dtype, polars.List):
            raise TypeError(f'velocities dtype must be List but is {velocities.dtype}')
        if not (velocities.list.len() == 2).all():
            list_lengths = velocities.list.len().unique().to_list()
            raise ValueError(f'velocities must be 2D list but list lengths are: {list_lengths}')
        velocities = numpy.vstack([velocities.list.get(0), velocities.list.get(1)]).transpose()
    velocities = numpy.array(velocities)
    _checks.check_shapes(velocities=velocities)

    if mu is not None:
        mu = numpy.array(mu)
    if sigma is not None:
        sigma = numpy.array(sigma)
    if init_state is not None:
        init_state = numpy.array(init_state)
    if transition_probabilities is not None:
        transition_probabilities = numpy.array(transition_probabilities)

    numeric_dtypes = polars.datatypes.FloatType, polars.datatypes.IntegerType
    if isinstance(timesteps, polars.Series):
        if not isinstance(timesteps.dtype, numeric_dtypes):
            raise TypeError(f'timesteps dtype must be float or int but is {timesteps.dtype}')
        timesteps = timesteps.to_numpy()
    elif timesteps is not None:
        timesteps = numpy.array(timesteps)
    else:
        timesteps = numpy.arange(len(velocities), dtype=numpy.int64)
    timesteps = numpy.array(timesteps).flatten()

    if not isinstance(minimum_duration, (int, numpy.integer)):
        raise TypeError(
            'minimum_duration must be of type int'
            f' but is of type {type(minimum_duration)}',
        )
    if minimum_duration <= 0:
        raise ValueError('minimum_duration must be greater than 0')

    _checks.check_is_length_matching(velocities=velocities, timesteps=timesteps)

    if mu is not None and mu.shape != (2,):
        raise ValueError(
            f'mu'
            f' must have shape (2,), but shapes are '
            f'{mu.shape}',
        )
    if sigma is not None and sigma.shape != (2,):
        raise ValueError(
            f'sigma'
            f' must have shape (2,), but shapes are '
            f'{sigma.shape}',
        )
    if init_state is not None and init_state.shape != (2,):
        raise ValueError(
            f'init_state'
            f' must have shape (2,), but shapes are '
            f'{init_state.shape}',
        )
    if transition_probabilities is not None and transition_probabilities.shape != (2, 2):
        raise ValueError(
            f'transition_probabilities'
            f' must have shape (2, 2), but shapes are '
            f'{transition_probabilities.shape}',
        )
    if transition_probabilities is not None:
        _validate_transition_probabilities(transition_probabilities)
    if init_state is not None:
        _validate_init_state(init_state)

    if hmm_parameters_dict is not None:
        hmm_parameters_dict = _validate_hmm_parameters_dict(hmm_parameters_dict)

    if not reestimation and verbose:
        warnings.warn(
            message=f"verbose is:{verbose} but reestimation is {reestimation},"
            f" verbose won't have any effect.",
            category=UserWarning,
            stacklevel=2,
        )

    # convert into velocities (1D velocities vector)

    velocities_1d = norm(velocities, axis=1)

    vel_mask = ~numpy.isnan(velocities_1d)

    # Without any valid sample there is nothing to decode; return no events instead
    # of failing on the data-driven initialization (matches ivt on all-nan input).
    if not numpy.any(vel_mask):
        return Events(name=name, onsets=numpy.array([]), offsets=numpy.array([]))

    start = numpy.argmax(vel_mask)
    end = len(velocities_1d) - numpy.argmax(vel_mask[::-1])

    velocities_1d = velocities_1d[start:end]

    vel_mask = vel_mask[start:end]

    timesteps_masked = timesteps[start:end]

    # compute HMM

    states = _compute_hmm(
        velocities=velocities_1d,
        verbose=verbose,
        reestimation=reestimation,
        reestimation_max_iters=reestimation_max_iters,
        mu=mu,
        sigma=sigma,
        init_state=init_state,
        transition_probabilities=transition_probabilities,
        velocities_mask=vel_mask,
        hmm_parameters_dict=hmm_parameters_dict,
    )

    # collapse states

    onsets_arr, offsets_arr = _collapse_states(
        states, timesteps=timesteps_masked, min_duration=minimum_duration,
    )

    # return event frame

    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)

    return events
