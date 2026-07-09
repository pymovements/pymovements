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

from pymovements._utils import _checks

from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events
from pymovements.gaze.transforms_numpy import norm 

def format_optimal_dict(opt: dict[str, Any]) -> dict[str, list[float] | list[list[float]]]:
    """
    Convert an optimization result dictionary into a JSON-serializable format.

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

    Args:
        opt: Dictionary containing optimization outputs. Values are expected
            to be NumPy arrays or array-like objects.

    Returns:
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
    out = {}
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


def emit_log_prob(
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    v: float,
    s: int,
) -> float:
    """
    Compute the log-probability of observing value `v` under a Gaussian emission model.

    This function evaluates the log-density of a univariate normal distribution
    parameterized by state-dependent mean (`mu`) and standard deviation (`sigma`),
    selecting parameters corresponding to state index `s`.

    A small numerical floor is applied to `sigma` to ensure stability.

    The computed quantity is:

        log p(v | s) = -0.5 * log(2πσ²) - (v - μ)² / (2σ²)

    Args:
        mu: Array of means for each hidden state. Shape: (num_states,).
            May be None if not used in a given context, but must be valid when accessed.
        sigma: Array of standard deviations for each hidden state. Shape: (num_states,).
            May be None if not used in a given context, but must be valid when accessed.
        v: Observed scalar value.
        s: Index of the hidden state used to select the corresponding (mu, sigma).

    Returns:
        The log-probability (float) of observing `v` given state `s`
        under a Gaussian emission model.
    """

    mu = mu[s]
    sigma = sigma[s]

    sigma = max(sigma, 1e-6)

    return -0.5 * numpy.log(2 * numpy.pi * sigma**2) - ((v - mu)**2) / (2 * sigma**2)


def log_sum_exp(
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


def baum_welch(
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
    """
    Estimate Hidden Markov Model parameters using the Baum-Welch algorithm.

    The Baum-Welch algorithm is an expectation-maximization (EM) algorithm used to
    find the maximum likelihood estimates of HMM parameters. This implementation
    handles partially observed velocity data through a masking mechanism.

    Parameters
    ----------
    states : int
        Number of hidden states in the HMM (M).

    mu : numpy.ndarray | None
        Initial means for the observation distributions (Gaussian emissions).
        Shape: (states,). If None, will be initialized during algorithm execution.

    sigma : numpy.ndarray | None
        Initial standard deviations for the observation distributions.
        Shape: (states,). If None, will be initialized during algorithm execution.

    init : numpy.ndarray | None
        Initial state probability distribution (log-space).
        Shape: (states,). If None, will be initialized from the forward-backward algorithm.

    trans : numpy.ndarray | None
        Initial state transition probability matrix (log-space).
        Shape: (states, states). trans[i, j] = log P(state_j | state_i).

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : array-like
        Boolean mask indicating which velocity observations are valid/observed.
        Same length as velocities. True indicates observed, False indicates missing.

    max_iters : int
        Maximum number of EM iterations to perform.

    epsilon : float, default=1e-4
        Convergence threshold. Algorithm stops when the relative change in
        log-likelihood between iterations is less than this value.

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
    """

    T = len(velocities)
    M = states

    prev_log_likelihood = -numpy.inf

    # main EM loop

    for _ in range(max_iters):

        # forward pass

        alpha = baum_forward(
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

        beta = baum_backward(
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
                            emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                            beta[t + 1, j],
                        )
                    else:
                        denom_terms.append(
                            alpha[t, i] +
                            trans[i, j] +
                            0.0 +
                            beta[t + 1, j],
                        )

            denom = log_sum_exp(numpy.array(denom_terms))

            for i in range(M):
                for j in range(M):
                    if velocities_mask[t + 1]:
                        num = (
                            alpha[t, i] +
                            trans[i, j] +
                            emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
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
        last = numpy.exp(last - log_sum_exp(last))
        gamma_full[:, -1] = last

        # m-step

        init = numpy.log(numpy.clip(gamma_full[:, 0], 1e-12, 1.0))

        # laplace smoothing for division by 0 errors
        eps = 1e-12
        for i in range(M):
            denom = numpy.sum(gamma_full[i, :-1])
            for j in range(M):
                numer = numpy.sum(xi[i, j, :])
                trans[i, j] = numpy.log((numer + eps) / (denom + eps * M))

        for j in range(M):

            mask = velocities_mask

            weights = gamma_full[j, mask]
            vals = numpy.asarray(velocities)[mask]

            total = numpy.sum(weights)

            mu[j] = numpy.sum(weights * vals) / total

            var = numpy.sum(weights * (vals - mu[j])**2) / total
            sigma[j] = numpy.sqrt(var)

        # compute log-likelihood for convergence check

        alpha_updated = baum_forward(
            mu=mu,
            sigma=sigma,
            trans=trans,
            init=init,
            velocities=velocities,
            velocities_mask=velocities_mask,
            T=T,
            M=M,
        )

        log_likelihood = log_sum_exp(alpha_updated[-1])

        if abs(log_likelihood - prev_log_likelihood) < epsilon:
            break

        prev_log_likelihood = log_likelihood

    return {'mu': mu, 'sigma': sigma, 'init': init, 'trans': trans}


def baum_forward(
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    init: numpy.ndarray | None,
    trans: numpy.ndarray | None,
    velocities: list[float] | numpy.ndarray,
    velocities_mask: numpy.ndarray,
    T: int,
    M: int,
) -> numpy.ndarray:
    """
    Compute forward probabilities (alpha) for a Hidden Markov Model.

    The forward algorithm computes the probability of being in each hidden state
    at each time step given the observed sequence up to that point. This implementation
    handles partially observed data through a masking mechanism and uses log-space
    computations for numerical stability.

    Parameters
    ----------
    mu : numpy.ndarray | None
        Means of the emission distributions (Gaussian) for each state.
        Shape: (M,). If None, emission probabilities are ignored (treated as log(1) = 0).

    sigma : numpy.ndarray | None
        Standard deviations of the emission distributions for each state.
        Shape: (M,). If None, emission probabilities are ignored.

    init : numpy.ndarray | None
        Initial state probability distribution (log-space).
        Shape: (M,). init[s] = log(P(state = s at time 0)).

    trans : numpy.ndarray | None
        State transition probability matrix (log-space).
        Shape: (M, M). trans[i, j] = log(P(state = j at time t | state = i at time t-1)).

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : array-like
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
    """
    alpha = numpy.full((T, M), -numpy.inf)

    # init step

    for s in range(M):
        if velocities_mask[0]:
            alpha[0, s] = init[s] + emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=s)
        else:
            alpha[0, s] = init[s] + 0

    # induction step

    for t in range(1, T):
        for j in range(M):
            terms = []
            for i in range(M):
                terms.append(alpha[t - 1, i] + trans[i, j])
            if velocities_mask[t]:
                alpha[t, j] = log_sum_exp(numpy.array(terms)) + \
                    emit_log_prob(mu=mu, sigma=sigma, v=velocities[t], s=j)
            else:
                alpha[t, j] = log_sum_exp(numpy.array(terms)) + \
                    0.0

    return alpha


def baum_backward(
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    trans: numpy.ndarray | None,
    velocities: list[float] | numpy.ndarray,
    velocities_mask: numpy.ndarray,
    T: int,
    M: int,
) -> numpy.ndarray:
    """
    Compute backward probabilities (beta) for a Hidden Markov Model.

    The backward algorithm computes the probability of the future observation sequence
    given that the system is in a particular state at a particular time. This implementation
    handles partially observed data through a masking mechanism and uses log-space
    computations for numerical stability.

    Parameters
    ----------
    mu : numpy.ndarray | None
        Means of the emission distributions (Gaussian) for each state.
        Shape: (M,). If None, emission probabilities are ignored (treated as log(1) = 0).

    sigma : numpy.ndarray | None
        Standard deviations of the emission distributions for each state.
        Shape: (M,). If None, emission probabilities are ignored.

    trans : numpy.ndarray | None
        State transition probability matrix (log-space).
        Shape: (M, M). trans[i, j] = log(P(state = j at time t+1 | state = i at time t)).

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : array-like
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

    """

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
                        emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                        beta[t + 1, j],
                    )
                else:
                    terms.append(
                        trans[i, j] +
                        0.0 +
                        beta[t + 1, j],
                    )

            beta[t, i] = log_sum_exp(numpy.array(terms))

    return beta


def viterbi(
    states: int,
    mu: numpy.ndarray | None,
    sigma: numpy.ndarray | None,
    init: numpy.ndarray | None,
    trans: numpy.ndarray | None,
    velocities: list[float] | numpy.ndarray,
    velocities_mask: numpy.ndarray,
) -> numpy.ndarray:
    """
    Find the most likely sequence of hidden states using the Viterbi algorithm.

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
        Shape: (states,). If None, emission probabilities are ignored (treated as log(1) = 0).

    sigma : numpy.ndarray | None
        Standard deviations of the emission distributions for each state.
        Shape: (states,). If None, emission probabilities are ignored.

    init : numpy.ndarray | None
        Initial state probability distribution (log-space).
        Shape: (states,). init[s] = log(P(state = s at time 0)).

    trans : numpy.ndarray | None
        State transition probability matrix (log-space).
        Shape: (states, states). trans[i, j] = log(P(state = j at time t | state = i at time t-1)).

    velocities : list[float] | numpy.ndarray
        Observation sequence of velocity measurements.
        Length: T (number of time steps).

    velocities_mask : list[bool]
        Boolean mask indicating which velocity observations are valid/observed.
        Length: T. True indicates observed, False indicates missing.

    Returns
    -------
    numpy.ndarray
        Most likely sequence of hidden states (Viterbi path).
        Shape: (T,), dtype=int. Each entry is a state index from 0 to states-1.

    """

    # init step

    T = len(velocities)

    prob = numpy.full((T, states), -numpy.inf)
    prev = numpy.zeros((T, states), dtype=int)

    for s in range(states):
        prob[0, s] = init[s] + emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=s)

    # main loop

    for t in range(1, T):
        for state1 in range(states):
            best_prob = -numpy.inf
            best_state = 0
            for state2 in range(states):
                if velocities_mask[t]:
                    new_prob = prob[t - 1, state2] + trans[state2, state1] + \
                        emit_log_prob(mu=mu, sigma=sigma, v=velocities[t], s=state1)
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


def collapse_states(
        states: numpy.ndarray,
        timesteps: numpy.ndarray,
        fixation_state: int = 0,
        min_duration: int = 0,

) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Extract contiguous fixation periods from a sequence of state labels.

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

    fixation_state : int, default=0
        The state label that represents fixation periods.
        All other states are ignored. Default is 0 (commonly used for fixation).

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


def compute_hmm(
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
    """
    Compute HMM state sequence for velocity data using optional parameter reestimation.

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

    velocities_mask : array-like
        Boolean mask indicating valid/observed velocity values.
        Shape: (T,). True for observed, False for missing/NaN values.

    hmm_parameters_dict : dict or None
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
            'mu': [numpy.percentile(velocities_for_init, 30), numpy.percentile(velocities_for_init, 80)],
            'sigma': [numpy.sqrt(numpy.var(velocities_for_init) / 2), numpy.sqrt(numpy.var(velocities_for_init))],
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

    _init = numpy.log(_init)
    _trans = numpy.log(_trans)

    # reestimate if needed

    if reestimation:
        optimal = baum_welch(
            states=2,
            mu=_mu,
            sigma=_sigma,
            init=_init,
            trans=_trans,
            velocities=velocities,
            velocities_mask=velocities_mask,
            max_iters=reestimation_max_iters,
        )

        # reorder to enforce states order (0 = fixation)

        order = numpy.argsort(optimal['mu'])

        optimal['mu'] = numpy.array(optimal['mu'])[order]
        optimal['sigma'] = numpy.array(optimal['sigma'])[order]
        optimal['init'] = numpy.array(optimal['init'])[order]
        optimal['trans'] = numpy.array(optimal['trans'][order])[:, order]

        _mu = optimal['mu']
        _sigma = optimal['sigma']
        _init = optimal['init']
        _trans = optimal['trans']

        if verbose:
            print(f"Optimal parameters found by reestimation are:\n{format_optimal_dict(optimal)}")

    # inference the hmm

    states = viterbi(
        states=2,
        mu=_mu,
        sigma=_sigma,
        init=_init,
        trans=_trans,
        velocities=velocities,
        velocities_mask=velocities_mask,
    )

    return states


@register_event_detection
def ihmm(
        velocities: list[list[float]] | list[tuple[float, float]] | numpy.ndarray,
        timesteps: list[int] | numpy.ndarray | None = None,
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
    """
    Detect fixation events from velocity data using an Independent Hidden Markov Model (IHMM).

    This function implements a 2-state HMM specifically designed for eye-tracking
    data to distinguish between fixations (state 0) and saccades (state 1). It
    processes velocity time series, estimates optimal parameters via Baum-Welch
    (optional), decodes the most likely state sequence using Viterbi, and collapses
    contiguous fixation periods into events.

    Parameters
    ----------
    velocities : list[list[float]] | list[tuple[float, float]] | numpy.ndarray
        Velocity data. Can be:
        - 2D array of shape (T, 2) containing x and y velocity components
        - 1D array of shape (T,) containing pre-computed velocity magnitudes
        - List of (vx, vy) tuples or lists
        Will be converted to velocity magnitudes via Euclidean norm.

    timesteps : list[int] | numpy.ndarray | None, default=None
        Timestamp indices for each velocity sample. Must be integers.
        If None, uses sequential indices (0, 1, 2, ..., T-1).

    minimum_duration: int
        Minimum fixation duration. The duration should be the same unit as the timesteps array.

    mu : list[float] | numpy.ndarray | None, default=None
        Mean velocity for each state (Gaussian emissions).
        Shape: (2,), typically [fixation_mean, saccade_mean].
        If None, uses data-driven defaults or hmm_parameters_dict.

    sigma : list[float] | numpy.ndarray | None, default=None
        Standard deviation of velocity for each state.
        Shape: (2,), typically [fixation_std, saccade_std].
        If None, uses data-driven defaults or hmm_parameters_dict.

    init_state : list[float] | numpy.ndarray | None, default=None
        Initial state probability distribution (linear scale).
        Shape: (2,), e.g., [0.5, 0.5]. Must sum to 1.
        If None, uses defaults or hmm_parameters_dict.

    transition_probabilities : list[list[float]] | numpy.ndarray | None, default=None
        State transition probability matrix (linear scale).
        Shape: (2, 2). Each row must sum to 1.
        If None, uses default matrix [[0.95, 0.05], [0.05, 0.95]].

    reestimation_max_iters : int, default=1000
        Maximum number of Baum-Welch EM iterations if reestimation=True.

    reestimation : bool, default=False
        If True, performs Baum-Welch reestimation to optimize HMM parameters
        before state decoding. Recommended for robust parameter estimation.

    verbose : bool, default=False
        If True, prints parameter values and reestimation progress.
        Only effective when reestimation=True.

    hmm_parameters_dict : dict | None, default=None
        Dictionary containing custom HMM parameters with keys:
        - 'mu': list of 2 means
        - 'sigma': list of 2 standard deviations
        - 'init': list of 2 initial probabilities
        - 'trans': 2x2 transition probability matrix
        Overridden by explicit mu, sigma, init_state, transition_probabilities.

    name : str, default='fixation'
        Name for the detected events. Appears in the returned Events object.

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

    Raises
    ------
    TypeError
        If timesteps contain non-integer values.
    ValueError
        If parameter shapes are incorrect (not (2,) or (2,2)).
    ValueError
        If transition_probabilities rows don't sum to 1.
    ValueError
        If hmm_parameters_dict has incorrect keys or shapes.
    """
    velocities = numpy.array(velocities)

    if mu is not None:
        mu = numpy.array(mu)
    if sigma is not None:
        sigma = numpy.array(sigma)
    if init_state is not None:
        init_state = numpy.array(init_state)
    if transition_probabilities is not None:
        transition_probabilities = numpy.array(transition_probabilities)

    _checks.check_shapes(velocities=velocities)

    if timesteps is None:
        timesteps = numpy.arange(len(velocities), dtype=numpy.int64)
    timesteps = numpy.array(timesteps).flatten()

    if minimum_duration <= 0:
        raise ValueError('minimum_duration must be greater than 0')
    if not isinstance(minimum_duration, int):
        raise TypeError(
            'minimum_duration must be of type int'
            f' but is of type {type(minimum_duration)}',
        )

    # check that timesteps are integers or are floats without a fractional part.
    timesteps_int = timesteps.astype(int)
    if numpy.any((timesteps - timesteps_int) != 0):
        raise TypeError('timesteps must be of type int')
    timesteps = timesteps_int

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
    if transition_probabilities is not None and numpy.sum(
            transition_probabilities[0],
    ) > 1 and numpy.sum(transition_probabilities[1]) > 1:
        sum0 = numpy.sum(transition_probabilities[0])
        sum1 = numpy.sum(transition_probabilities[1])
        raise ValueError(
            f'transition_probabilities values must sum up to one for each state '
            f'but instead are {sum0} and {sum1}',
        )

    if hmm_parameters_dict is not None:

        if list(hmm_parameters_dict.keys()) != ['mu', 'sigma', 'init', 'trans']:
            raise ValueError(
                f'hmm_parameters_dict'
                f' should have fields {["mu", "sigma", "init", "trans"]} but instead has '
                f'{hmm_parameters_dict.keys()}',
            )

        hmm_parameters_dict['mu'] = numpy.array(hmm_parameters_dict['mu'])
        hmm_parameters_dict['sigma'] = numpy.array(hmm_parameters_dict['sigma'])
        hmm_parameters_dict['init'] = numpy.array(hmm_parameters_dict['init'])
        hmm_parameters_dict['trans'] = numpy.array(hmm_parameters_dict['trans'])

        if hmm_parameters_dict['mu'] is not None and hmm_parameters_dict['mu'].shape != (2,):
            raise ValueError(
                f'mu'
                f' must have shape (2,), but shapes are '
                f'{hmm_parameters_dict["mu"].shape}',
            )
        if hmm_parameters_dict['sigma'] is not None and hmm_parameters_dict['sigma'].shape != (2,):
            raise ValueError(
                f'sigma'
                f' must have shape (2,), but shapes are '
                f'{hmm_parameters_dict["sigma"].shape}',
            )
        if hmm_parameters_dict['init'] is not None and hmm_parameters_dict['init'].shape != (2,):
            raise ValueError(
                f'init_state'
                f' must have shape (2,), but shapes are '
                f'{hmm_parameters_dict["init"].shape}',
            )
        if hmm_parameters_dict['trans'] is not None and hmm_parameters_dict['trans'].shape != (
                2, 2,
        ):
            raise ValueError(
                f'transition_probabilities'
                f' must have shape (2, 2), but shapes are '
                f'{hmm_parameters_dict["trans"].shape}',
            )

    if not reestimation and verbose:
        warnings.warn(
            message=f"verbose is:{verbose} but reestimation is {reestimation}, verbose won't have any effect.",
        )

    # convert into velocities (1D velocities vector)
    

    velocities_1d = norm(velocities, axis=1)

    vel_mask = ~numpy.isnan(velocities_1d)
    cW = 0
    for val in vel_mask:
        if val:
            pass
        else:
            cW += 1

    start = numpy.argmax(vel_mask)
    end = len(velocities_1d) - numpy.argmax(vel_mask[::-1])

    velocities_1d = velocities_1d[start:end]

    vel_mask = vel_mask[start:end]

    timesteps_masked = timesteps[start:end]

    # compute HMM

    states = compute_hmm(
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

    onsets_arr, offsets_arr = collapse_states(
        states, timesteps=timesteps_masked, min_duration=minimum_duration,
    )

    # return event frame

    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)

    return events
