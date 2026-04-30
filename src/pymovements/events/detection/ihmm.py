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
from __future__ import annotations

import numpy as np

from pymovements._utils import _checks
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events


def emit_log_prob(
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    v: float,
    s: int,
) -> float:
    """Compute the log-probability of an observation given a state.

    Parameters
    ----------
    mu : np.ndarray | None
        shape (2,)
        Initial means of emission distributions.
    sigma : np.ndarray | None
        shape (2,)
        Initial standard deviations of emission distributions.
    v : float
        Observed value (e.g., velocity).
    s : int
        State index.

    Returns
    -------
    float
        Log-probability of observing v in state s.
    """
    mu = mu[s]
    sigma = sigma[s]

    sigma = max(sigma, 1e-6)

    return -0.5 * np.log(2 * np.pi * sigma**2) - ((v - mu)**2) / (2 * sigma**2)


def log_sum_exp(
    arr: np.ndarray,
) -> float:
    """Compute log-sum-exp.

    Parameters
    ----------
    arr : np.ndarray
        Input array of log-values.

    Returns
    -------
    float
        Logarithm of the summed exponentials.
    """
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))


def baum_welch(
    states: int,
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    init: np.ndarray | None,
    trans: np.ndarray | None,
    velocities: list[float] | np.ndarray,
    max_iters: int,
    epsilon: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Estimate HMM parameters using the Baum-Welch algorithm.

    This is an Expectation-Maximization (EM) procedure that iteratively updates
    the model parameters (initial state, transition probabilities, and emission
    distributions) to maximize the likelihood of the observed data.

    Parameters
    ----------
    states : int
            Number of hidden states in the model.
    mu : np.ndarray | None
        shape (2,)
        Initial means of emission distributions.
    sigma : np.ndarray | None
        shape (2,)
        Initial standard deviations of emission distributions.
    init : np.ndarray | None
        shape (2,)
        Initial state probabilities.
    trans : np.ndarray | None
        shape (2, 2)
        State transition probabilities.
    velocities : list[float] | np.ndarray
        shape (T,)
        Sequence of observed values.
    max_iters : int
        Maximum number of EM iterations.
    epsilon : float
        Convergence threshold for change in log-likelihood.
        (default: 1e-4)

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing updated parameters:
        - "mu": means of emission distributions
        - "sigma": standard deviations of emission distributions
        - "init": log initial state probabilities
        - "trans": log transition matrix
    """

    T = len(velocities)
    M = states

    prev_log_likelihood = -np.inf

    for _ in range(max_iters):

        alpha = baum_forward(
            mu=mu,
            sigma=sigma,
            trans=trans,
            init=init,
            velocities=velocities,
            T=T,
            M=M,
        )

        beta = baum_backward(mu=mu, sigma=sigma, trans=trans, velocities=velocities, T=T, M=M)

        xi = np.zeros((M, M, T - 1))

        for t in range(T - 1):
            denom_terms = []

            for i in range(M):
                for j in range(M):
                    denom_terms.append(
                        alpha[t, i] +
                        trans[i, j] +
                        emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                        beta[t + 1, j],
                    )

            denom = log_sum_exp(np.array(denom_terms))

            for i in range(M):
                for j in range(M):
                    num = (
                        alpha[t, i] +
                        trans[i, j] +
                        emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                        beta[t + 1, j]
                    )
                    xi[i, j, t] = np.exp(num - denom)

        gamma = np.sum(xi, axis=1)

        gamma_full = np.zeros((M, T))
        gamma_full[:, :-1] = gamma

        last = alpha[T - 1] + beta[T - 1]
        last = np.exp(last - log_sum_exp(last))
        gamma_full[:, -1] = last

        # init = np.log(gamma_full[:, 0])
        init = np.log(np.clip(gamma_full[:, 0], 1e-12, 1.0))

        for i in range(M):
            denom = np.sum(gamma_full[i, :-1])
            for j in range(M):
                numer = np.sum(xi[i, j, :])
                trans[i, j] = np.log(numer / denom)

        for j in range(M):
            weights = gamma_full[j, :]
            total = np.sum(weights)

            mu[j] = np.sum(weights * velocities) / total

            var = np.sum(weights * (velocities - mu[j])**2) / total
            sigma[j] = np.sqrt(var)

        alpha_updated = baum_forward(
            mu=mu,
            sigma=sigma,
            trans=trans,
            init=init,
            velocities=velocities,
            T=T,
            M=M,
        )

        log_likelihood = log_sum_exp(alpha_updated[-1])

        if abs(log_likelihood - prev_log_likelihood) < epsilon:
            break

        prev_log_likelihood = log_likelihood

    return {'mu': mu, 'sigma': sigma, 'init': init, 'trans': trans}


def baum_forward(
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    init: np.ndarray | None,
    trans: np.ndarray | None,
    velocities: list[float] | np.ndarray,
    T: int,
    M: int,
) -> np.ndarray:
    """Compute forward probabilities (alpha) in log-space.

    The forward algorithm calculates the probability of observing the sequence
    up to time t and being in state j at time t.

    Parameters
    ----------
    mu : np.ndarray | None
        shape (2,)
        Initial means of emission distributions.
    sigma : np.ndarray | None
        shape (2,)
        Initial standard deviations of emission distributions.
    init : np.ndarray | None
        shape (2,)
        Initial state probabilities.
    trans : np.ndarray | None
        shape (2, 2)
        State transition probabilities.
    velocities : list[float] | np.ndarray
        shape (T,)
        Sequence of observed values.
    T : int
        Length of the sequence.
    M : int
        Number of states.

    Returns
    -------
    np.ndarray
        shape (T, M)
        Log forward probabilities.
    """

    alpha = np.full((T, M), -np.inf)

    for s in range(M):
        alpha[0, s] = init[s] + emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=s)

    for t in range(1, T):
        for j in range(M):
            terms = []
            for i in range(M):
                terms.append(alpha[t - 1, i] + trans[i, j])
            alpha[t, j] = log_sum_exp(np.array(terms)) + \
                emit_log_prob(mu=mu, sigma=sigma, v=velocities[t], s=j)

    return alpha


def baum_backward(
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    trans: np.ndarray | None,
    velocities: list[float] | np.ndarray,
    T: int,
    M: int,
) -> np.ndarray:
    """Compute backward probabilities (beta) in log-space.

    The backward algorithm calculates the probability of observing the future
    sequence from time t+1 onward given state i at time t.

    Parameters
    ----------
    mu : np.ndarray | None
        shape (2,)
        Initial means of emission distributions.
    sigma : np.ndarray | None
        shape (2,)
        Initial standard deviations of emission distributions.
    trans : np.ndarray | None
        shape (2, 2)
        State transition probabilities.
    velocities : list[float] | np.ndarray
        shape (T,)
        Sequence of observed values.
    T : int
        Length of the sequence.
    M : int
        Number of states.

    Returns
    -------
    np.ndarray
        shape (T, M)
        Log backward probabilities.
    """

    beta = np.full((T, M), -np.inf)

    beta[T - 1, :] = 0

    for t in range(T - 2, -1, -1):
        for i in range(M):
            terms = []
            for j in range(M):
                terms.append(
                    trans[i, j] +
                    emit_log_prob(mu=mu, sigma=sigma, v=velocities[t + 1], s=j) +
                    beta[t + 1, j],
                )
            beta[t, i] = log_sum_exp(np.array(terms))

    return beta


def viterbi(
    states: int,
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    init: np.ndarray | None,
    trans: np.ndarray | None,
    velocities: list[float] | np.ndarray,
) -> np.ndarray:
    """Compute the most likely state sequence using the Viterbi algorithm.

    This dynamic programming algorithm finds the sequence of hidden states
    that maximizes the joint probability of the observations and the states.

    Parameters
    ----------
    states : int
            Number of hidden states in the model.
    mu : np.ndarray | None
        shape (2,)
        Initial means of emission distributions.
    sigma : np.ndarray | None
        shape (2,)
        Initial standard deviations of emission distributions.
    init : np.ndarray | None
        shape (2,)
        Initial state probabilities.
    trans : np.ndarray | None
        shape (2, 2)
        State transition probabilities.
    velocities : list[float] | np.ndarray
        shape (T,)
        Sequence of observed values.

    Returns
    -------
    np.ndarray
        shape (T,)
        Most likely sequence of state indices.
    """

    # init step

    T = len(velocities)

    prob = np.full((T, states), -np.inf)
    prev = np.zeros((T, states), dtype=int)

    for s in range(states):
        prob[0, s] = init[s] + emit_log_prob(mu=mu, sigma=sigma, v=velocities[0], s=s)

    # main loop

    for t in range(1, T):
        for state1 in range(states):
            best_prob = -np.inf
            best_state = 0
            for state2 in range(states):
                new_prob = prob[t - 1, state2] + trans[state2, state1] + \
                    emit_log_prob(mu=mu, sigma=sigma, v=velocities[t], s=state1)
                if new_prob > best_prob:
                    best_prob = new_prob
                    best_state = state2
            prob[t, state1] = best_prob
            prev[t, state1] = best_state

    # backtrack

    path = np.zeros(T, dtype=int)

    path[T - 1] = np.argmax(prob[T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = prev[t + 1, path[t + 1]]

    return path


def collapse_states(
        states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a sequence of HMM states into event onsets and offsets.

    This function assumes a binary state model where state `0` represents
    the event of interest (e.g., fixation) and state `1` represents the
    other class (e.g., saccade). It extracts contiguous segments of state 0.

    Parameters
    ----------
    states : np.ndarray
        shape (T,)
        Sequence of inferred state indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays:
        - onsets: indices where state 0 segments start
        - offsets: indices where state 0 segments end
    """

    onsets_arr = []
    offsets_arr = []

    prev_state = states[0]

    if prev_state == 0:
        onsets_arr.append(0)

    for i, state in enumerate(states[1:], start=1):

        if state == 0:
            if prev_state != 0:
                onsets_arr.append(i)
        else:
            if prev_state == 0:
                offsets_arr.append(i - 1)

        prev_state = state

    if prev_state == 0:
        offsets_arr.append(len(states) - 1)

    onsets_arr = np.array(onsets_arr)
    offsets_arr = np.array(offsets_arr)

    return onsets_arr, offsets_arr


def compute_hmm(
    velocities: np.ndarray,
    verbose: bool,
    initialization: str | None,
    reestimation_max_iters: int,
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    init_state: np.ndarray | None,
    transition_probabilities: np.ndarray | None,
) -> np.ndarray:
    """Run HMM parameter setup, optional Baum-Welch reestimation, and Viterbi decoding.

    This function initializes HMM parameters (either from defaults or user input),
    optionally refines them using the Baum-Welch algorithm, and then computes the
    most likely hidden state sequence using the Viterbi algorithm.

    Parameters
    ----------
    velocities : np.ndarray
        shape (T,)
        Sequence of observed velocities.
    verbose : bool
        If True, prints parameters after reestimation.
    initialization : str | None
        Initialization mode:
        - None: use provided parameters or defaults
        - 'default': use default parameters
        - 'reestimation': use defaults and apply Baum-Welch
    reestimation_max_iters : int
        Maximum number of iterations for Baum-Welch.
    mu : np.ndarray | None
        shape (2,)
        Initial means of emission distributions.
    sigma : np.ndarray | None
        shape (2,)
        Initial standard deviations of emission distributions.
    init_state : np.ndarray | None
        shape (2,)
        Initial state probabilities.
    transition_probabilities : np.ndarray | None
        shape (2, 2)
        State transition probabilities.

    Returns
    -------
    np.ndarray
        shape (T,)
        Most likely sequence of hidden states.
    """
    reestimate = False

    defaults = {
        # DATA BASED init  #[1.0, 10.0],
        'mu': [np.percentile(velocities, 30), np.percentile(velocities, 80)],
        # DATA BASED init   #[1.0, 1.0],
        'sigma': [np.sqrt(np.var(velocities) / 2), np.sqrt(np.var(velocities))],
        'init': [0.5, 0.5],  # dummy average values should be fine for long sequences
        'trans': [[0.95, 0.05], [0.05, 0.95]],  # based on Salvucci's paper diagram
    }

    match initialization:
        case 'reestimation':
            reestimate = True
            _mu = defaults['mu']
            _sigma = defaults['sigma']
            _init = defaults['init']
            _trans = defaults['trans']
        case 'default':
            _mu = defaults['mu']
            _sigma = defaults['sigma']
            _init = defaults['init']
            _trans = defaults['trans']
        case _:
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

    _init = np.log(_init)
    _trans = np.log(_trans)

    if reestimate:
        optimal = baum_welch(
            states=2,
            mu=_mu,
            sigma=_sigma,
            init=_init,
            trans=_trans,
            velocities=velocities,
            max_iters=reestimation_max_iters,
        )
        _mu = optimal['mu']
        _sigma = optimal['sigma']
        _init = optimal['init']
        _trans = optimal['trans']

        if verbose:
            print(f"Optimal parameters found by reestimation are:\n{optimal}")

    # inference the hmm

    states = viterbi(
        states=2,
        mu=_mu,
        sigma=_sigma,
        init=_init,
        trans=_trans,
        velocities=velocities,
    )

    return states


@register_event_detection
def ihmm(
        velocities: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        mu: list[float] | np.ndarray | None = None,
        sigma: list[float] | np.ndarray | None = None,
        init_state: list[float] | np.ndarray | None = None,
        transition_probabilities: list[list[float]] | np.ndarray | None = None,
        reestimation_max_iters: int = 1000,
        initialization: str | None = None,
        verbose: bool = False,
        name: str = 'fixation',
) -> Events:
    """
    Fixation identification based on a two state Hidden Markov Model.

    The algorithm models eye movements using a two-state Hidden Markov Model (HMM).
    One state represents fixations (low velocities), and the other represents saccades (high velocities).
    It analyzes the sequence of velocities and uses dynamic programming (Viterbi decoding)
    to assign each point to the most likely state. This results in classifying every point as either a fixation or a saccade.

    Parameters
    ----------
    velocities: list[list[float]] | list[tuple[float, float]] | numpy.ndarray | polars.Series
        shape (N, 2)
        Corresponding continuous 2D velocity time series.
    timesteps: list[int] | np.ndarray | None
        shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed. (default: None)
        (default: None)
    mu: list[float] | np.ndarray | None = None
        shape (2,)
        Array of means for the fixations distribution and saccades distribution.
        (default: None)
    sigma: list[float] | np.ndarray | None = None
        shape (2,)
        Array of standard deviations for the fixations distribution and saccades distribution.
        (default: None)
    init_state: list[float] | np.ndarray | None = None
        shape (2,)
        Initial probability of starting in each state.
        (default: None)
    transition_probabilities:
        shape (2, 2)
        Probabilities to change from a state to another.
        (default: None)
    reestimation_max_iters: int
        Number of maximum iterations for the Baum-Welch reestimation algorithm.
        (default: 100)
    initialization: str
        Initialization mode, default or None for default parameters and 'reestimation' for Baum-Welch reestimation.
        (default: None)
    name: str
        Name for detected events in Events. (default: 'fixation')

    Returns
    -------
    Events
        A dataframe with detected fixations as rows.

    Raises
    ------
    ValueError
        If velocities is None
        If velocities does not have shape (N, 2)
        If mu is not shaped (2,).
        If sigma is not shaped (2,).
        If init_state is not shaped (2,).
        If transition_probabilities is not shaped (2, 2).
        If transition_probabilities do not sum up to 1.

    Examples
    --------
    >>> import numpy as np
    >>> from pymovements.synthetic import step_function
    >>> from pymovements.gaze import from_numpy
    >>> from pymovements.events.detection import ihmm

    Create synthetic gaze data.
    >>> positions = step_function(
        ...    length=200, steps=[2, 5, 9, 111, 150],
        ...    values=[(1., 2.), (2., 3.), (3., 4.), (1., 1.), (2., 2.)],
        ...    start_value=(0., 0.))
    >>> positions.shape
    shape: (200, 2)

    Detect fixations with default parameters.
    >>> events = ihmm(positions)
    >>> events
    shape: (3, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 9     ┆ 109    ┆ 100      │
    │ fixation ┆ 111   ┆ 148    ┆ 37       │
    │ fixation ┆ 150   ┆ 198    ┆ 48       │
    └──────────┴───────┴────────┴──────────┘

    The IHMM algorithm can also be used with a Gaze object using the detect() method.

    Initialize Gaze object.
    >>> gaze = from_numpy(
         position=positions.T,
         time=np.arange(len(positions)),
      )
    >>> gaze
    shape: (200, 2)
    ┌──────┬────────────┐
    │ time ┆ position   │
    │ ---  ┆ ---        │
    │ i64  ┆ list[f64]  │
    ╞══════╪════════════╡
    │ 0    ┆ [0.0, 0.0] │
    │ 1    ┆ [0.0, 0.0] │
    │ 2    ┆ [1.0, 2.0] │
    │ 3    ┆ [1.0, 2.0] │
    │ 4    ┆ [1.0, 2.0] │
    │ …    ┆ …          │
    │ 195  ┆ [2.0, 2.0] │
    │ 196  ┆ [2.0, 2.0] │
    │ 197  ┆ [2.0, 2.0] │
    │ 198  ┆ [2.0, 2.0] │
    │ 199  ┆ [2.0, 2.0] │
    └──────┴────────────┘

    Detect fixations with Baum-Welch reestimation.
    >>> gaze.detect('ihmm', intialization="reestimation")
    >>> gaze.events
    shape: (6, 4)
    ┌──────────┬───────┬────────┬──────────┐
    │ name     ┆ onset ┆ offset ┆ duration │
    │ ---      ┆ ---   ┆ ---    ┆ ---      │
    │ str      ┆ i64   ┆ i64    ┆ i64      │
    ╞══════════╪═══════╪════════╪══════════╡
    │ fixation ┆ 9     ┆ 109    ┆ 100      │
    │ fixation ┆ 111   ┆ 148    ┆ 37       │
    │ fixation ┆ 150   ┆ 198    ┆ 48       │
    └──────────┴───────┴────────┴──────────┘
    """

    velocities = np.array(velocities)

    if mu is not None:
        mu = np.array(mu)
    if sigma is not None:
        sigma = np.array(sigma)
    if init_state is not None:
        init_state = np.array(init_state)
    if transition_probabilities is not None:
        transition_probabilities = np.array(transition_probabilities)

    _checks.check_shapes(velocities=velocities)

    if timesteps is None:
        timesteps = np.arange(len(velocities), dtype=np.int64)
    timesteps = np.array(timesteps).flatten()

    # Check that timesteps are integers or are floats without a fractional part.
    timesteps_int = timesteps.astype(int)
    if np.any((timesteps - timesteps_int) != 0):
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
    if transition_probabilities is not None and np.sum(
            transition_probabilities[0],
    ) > 1 and np.sum(transition_probabilities[1]) > 1:
        raise ValueError(
            f'transition_probabilities'
            f' values must sum up to one for each state but instead are '
            f'{np.sum(transition_probabilities[0])} and {np.sum(transition_probabilities[1])}',
        )

    if initialization is not None and initialization != 'reestimation' and initialization != 'default':
        raise ValueError(
            f'initialization'
            f' must either be None "reestimation" or "default" and instead is '
            f'{initialization}',
        )

    # convert into velocities (1D velocities vector)

    velocities_1d = np.array(
        list(map(lambda x: np.sqrt(x[0]**2 + x[1]**2), velocities)),
    )

    velocities_1d = np.nan_to_num(velocities_1d, nan=0.0)

    # compute HMM

    states = compute_hmm(
        velocities=velocities_1d,
        verbose=verbose,
        initialization=initialization,
        reestimation_max_iters=reestimation_max_iters,
        mu=mu,
        sigma=sigma,
        init_state=init_state,
        transition_probabilities=transition_probabilities,
    )

    # collapse states

    onsets_arr, offsets_arr = collapse_states(states)

    # return event frame

    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)

    return events
