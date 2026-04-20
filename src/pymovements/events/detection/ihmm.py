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


class HMM:

    # TODO: some string representation method? To inspect states and so on

    def __init__(
            self,
            states: int,
            mu: list[float] | np.ndarray,
            sigma: list[float] | np.ndarray,
            initial_state: list[float] | np.ndarray,
            transition_matrix: list[list[float]] | np.ndarray,
    ) -> None:
        """Initialize a Hidden Markov Model with Gaussian emissions.

        The model uses log-space for numerical stability. Each state is
        associated with a Gaussian distribution defined by its mean and standard
        deviation, and transitions between states are governed by a transition matrix.

        Parameters
        ----------
        states : int
            Number of hidden states in the model.
        mu : list[float] | np.ndarray
            shape (states,)
            Mean of the emission distribution for each state.
        sigma : list[float] | np.ndarray
            shape (states,)
            Standard deviation of the emission distribution for each state.
        initial_state : list[float] | np.ndarray
            shape (states,)
            Initial probability distribution over states. Must sum to 1.
        transition_matrix : list[list[float]] | np.ndarray
            shape (states, states)
            State transition probability matrix.

        Returns
        -------
        None
        """
        self.states = states

        self.init = np.log(initial_state)

        self.mu = mu

        self.sigma = sigma

        self.trans = np.log(transition_matrix)

        return

    def emit_log_prob(
            self,
            v: float,
            s: int,
    ) -> float:
        """Compute the log-probability of an observation given a state.

        Parameters
        ----------
        v : float
            Observed value (e.g., velocity).
        s : int
            State index.

        Returns
        -------
        float
            Log-probability of observing v in state s.
        """
        mu = self.mu[s]
        sigma = self.sigma[s]

        sigma = max(sigma, 1e-6)

        return -0.5 * np.log(2 * np.pi * sigma**2) - ((v - mu)**2) / (2 * sigma**2)

    def log_sum_exp(
            self,
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
            self,
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
        M = self.states

        prev_log_likelihood = -np.inf

        for _ in range(max_iters):

            alpha = self.baum_forward(velocities, T, M)

            beta = self.baum_backward(velocities, T, M)

            xi = np.zeros((M, M, T - 1))

            for t in range(T - 1):
                denom_terms = []

                for i in range(M):
                    for j in range(M):
                        denom_terms.append(
                            alpha[t, i] +
                            self.trans[i, j] +
                            self.emit_log_prob(velocities[t + 1], j) +
                            beta[t + 1, j],
                        )

                denom = self.log_sum_exp(np.array(denom_terms))

                for i in range(M):
                    for j in range(M):
                        num = (
                            alpha[t, i] +
                            self.trans[i, j] +
                            self.emit_log_prob(velocities[t + 1], j) +
                            beta[t + 1, j]
                        )
                        xi[i, j, t] = np.exp(num - denom)

            gamma = np.sum(xi, axis=1)

            gamma_full = np.zeros((M, T))
            gamma_full[:, :-1] = gamma

            last = alpha[T - 1] + beta[T - 1]
            last = np.exp(last - self.log_sum_exp(last))
            gamma_full[:, -1] = last

            self.init = np.log(gamma_full[:, 0])

            for i in range(M):
                denom = np.sum(gamma_full[i, :-1])
                for j in range(M):
                    numer = np.sum(xi[i, j, :])
                    self.trans[i, j] = np.log(numer / denom)

            for j in range(M):
                weights = gamma_full[j, :]
                total = np.sum(weights)

                self.mu[j] = np.sum(weights * velocities) / total

                var = np.sum(weights * (velocities - self.mu[j])**2) / total
                self.sigma[j] = np.sqrt(var)

            alpha_updated = self.baum_forward(velocities, T, M)

            log_likelihood = self.log_sum_exp(alpha_updated[-1])

            if abs(log_likelihood - prev_log_likelihood) < epsilon:
                break

            prev_log_likelihood = log_likelihood

        return {'mu': self.mu, 'sigma': self.sigma, 'init': self.init, 'trans': self.trans}

    def baum_forward(
            self,
            velocities: list[float] | np.ndarray,
            T: int,
            M: int,
    ) -> np.ndarray:
        """Compute forward probabilities (alpha) in log-space.

        The forward algorithm calculates the probability of observing the sequence
        up to time t and being in state j at time t.

        Parameters
        ----------
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
            alpha[0, s] = self.init[s] + self.emit_log_prob(velocities[0], s)

        for t in range(1, T):
            for j in range(M):
                terms = []
                for i in range(M):
                    terms.append(alpha[t - 1, i] + self.trans[i, j])
                alpha[t, j] = self.log_sum_exp(np.array(terms)) + \
                    self.emit_log_prob(velocities[t], j)

        return alpha

    def baum_backward(
            self,
            velocities: list[float] | np.ndarray,
            T: int,
            M: int,
    ) -> np.ndarray:
        """Compute backward probabilities (beta) in log-space.

        The backward algorithm calculates the probability of observing the future
        sequence from time t+1 onward given state i at time t.

        Parameters
        ----------
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
                        self.trans[i, j] +
                        self.emit_log_prob(velocities[t + 1], j) +
                        beta[t + 1, j],
                    )
                beta[t, i] = self.log_sum_exp(np.array(terms))

        return beta

    def viterbi(
            self,
            velocities: list[float] | np.ndarray,
    ) -> np.ndarray:
        """Compute the most likely state sequence using the Viterbi algorithm.

        This dynamic programming algorithm finds the sequence of hidden states
        that maximizes the joint probability of the observations and the states.

        Parameters
        ----------
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

        prob = np.full((T, self.states), -np.inf)
        prev = np.zeros((T, self.states), dtype=int)

        for s in range(self.states):
            prob[0, s] = self.init[s] + self.emit_log_prob(velocities[0], s)

        # main loop

        for t in range(1, T):
            for state1 in range(self.states):
                best_prob = -np.inf
                best_state = 0
                for state2 in range(self.states):
                    new_prob = prob[t - 1, state2] + self.trans[state2, state1] + \
                        self.emit_log_prob(velocities[t], state1)
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


@register_event_detection
def ihmm(
        positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        mu: list[float] | np.ndarray | None = None,
        sigma: list[float] | np.ndarray | None = None,
        init_state: list[float] | np.ndarray | None = None,
        transition_probabilities: list[list[float]] | np.ndarray | None = None,
        reestimation_max_iters: int = 100,
        initialization: str | None = None,
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
    positions: list[list[float]] | list[tuple[float, float]] | np.ndarray
        shape (N, 2)
        Continuous 2D position time series
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
        If positions is not shaped (N, 2)
        If mu is not shaped (2,)
        If sigma is not shaped (2,)
        If init_state is not shaped (2,)
        If transition_probabilities is not shaped (2, 2)

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

    positions = np.array(positions)

    if mu is not None:
        mu = np.array(mu)
    if sigma is not None:
        sigma = np.array(sigma)
    if init_state is not None:
        init_state = np.array(init_state)
    if transition_probabilities is not None:
        transition_probabilities = np.array(transition_probabilities)

    _checks.check_shapes(positions=positions)

    if timesteps is None:
        timesteps = np.arange(len(positions), dtype=np.int64)
    timesteps = np.array(timesteps).flatten()

    # Check that timesteps are integers or are floats without a fractional part.
    timesteps_int = timesteps.astype(int)
    if np.any((timesteps - timesteps_int) != 0):
        raise TypeError('timesteps must be of type int')
    timesteps = timesteps_int

    _checks.check_is_length_matching(positions=positions, timesteps=timesteps)

    # DONE # TODO: Implement other dimension checks for inputs

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
            transition_probabilities[0]) > 1 and np.sum(transition_probabilities[1]) > 1:
        raise ValueError(
            f'transition_probabilities'
            f' values must sum up to one for each state but instead are '
            f'{np.sum(transition_probabilities[0])} and {np.sum(transition_probabilities[1])}',
        )

    # convert into velocities (1D velocities vector)

    # TODO: Optimize, maybe implement different vel algorithms/connect to
    # pos2vel method/make use of the velocity column if present

    velocities = []

    for ind in range(len(positions) - 1):

        i = ind + 1
        x_i = positions[i - 1][0]
        x_i_1 = positions[i][0]
        y_i = positions[i - 1][1]
        y_i_1 = positions[i][1]
        t_i = timesteps[i - 1]
        t_i_1 = timesteps[i]

        dt = t_i_1 - t_i

        if dt == 0:
            v_i = 0
        else:
            v_i = np.sqrt((x_i_1 - x_i)**2 + (y_i_1 - y_i)**2) / dt

        velocities.append(v_i)

    velocities = np.array(velocities)

    velocities = np.nan_to_num(velocities, nan=0.0)  # maybe should be average?

    # Init 2 state HMM

    defaults = {
        # DATA BASED init  #[1.0, 10.0],
        'mu': [np.percentile(velocities, 30), np.percentile(velocities, 80)],
        # DATA BASED init   #[1.0, 1.0],
        'sigma': [np.sqrt(np.var(velocities) / 2), np.sqrt(np.var(velocities))],
        'init': [0.5, 0.5],  # dummy average values should be fine for long sequences
        'trans': [[0.95, 0.05], [0.05, 0.95]],  # based on Salvucci's paper diagram
    }

    reestimate = False

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

    hmm = HMM(states=2, mu=_mu, sigma=_sigma, initial_state=_init, transition_matrix=_trans)

    if reestimate:
        optimal = hmm.baum_welch(velocities=velocities, max_iters=reestimation_max_iters)

    # inference the hmm

    states = hmm.viterbi(velocities=velocities)

    # collapse states

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

    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)

    return events
