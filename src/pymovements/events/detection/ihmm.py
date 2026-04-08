from __future__ import annotations

import numpy as np

from pymovements._utils import _checks
from pymovements.events._utils._filters import events_split_nans
from pymovements.events._utils._filters import filter_candidates_remove_nans
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events


class HMM:

    # TODO: some string representation method? To inspect states and so on

    def __init__(self,states,mu,sigma,initial_state,transition_matrix):

        # DONE # TODO: implement different initializations outside/inside

        self.states = states

        self.init = np.log(initial_state)

        self.mu=mu

        self.sigma = sigma

        self.trans = np.log(transition_matrix)
        
        return
    
    def emit_log_prob(self, v, s):
        mu = self.mu[s]
        sigma = self.sigma[s]
        return -0.5 * np.log(2*np.pi*sigma**2) - ((v - mu)**2) / (2*sigma**2)
    
    def log_sum_exp(self,arr):
        m = np.max(arr)
        return m + np.log(np.sum(np.exp(arr - m)))

    def baum_welch(self,velocities,n_iter=100):

        T = len(velocities)
        M = self.states

        # TODO: Implement convergence instead of iters

        for _ in range(n_iter):

            alpha = self.baum_forward(velocities,T,M)
            beta = self.baum_backward(velocities,T,M)

            xi = np.zeros((M, M, T-1))

            for t in range(T-1):
                denomTerms = []

                for i in range(M):
                    for j in range(M):
                        denomTerms.append(
                            alpha[t, i] +
                            self.trans[i, j] +
                            self.emit_log_prob(velocities[t+1], j) +
                            beta[t+1, j]
                        )

                denom = self.log_sum_exp(np.array(denomTerms))

                for i in range(M):
                    for j in range(M):
                        num = (
                            alpha[t, i] +
                            self.trans[i, j] +
                            self.emit_log_prob(velocities[t+1], j) +
                            beta[t+1, j]
                        )
                        xi[i, j, t] = np.exp(num - denom)

           
            gamma = np.sum(xi, axis=1) 

            gammaFull = np.zeros((M, T))
            gammaFull[:, :-1] = gamma

            last = alpha[T-1] + beta[T-1]
            last = np.exp(last - self.log_sum_exp(last))
            gammaFull[:, -1] = last

          
            self.init = np.log(gammaFull[:, 0])

            for i in range(M):
                denom = np.sum(gammaFull[i, :-1])
                for j in range(M):
                    numer = np.sum(xi[i, j, :])
                    self.trans[i, j] = np.log(numer / denom)

        
            for j in range(M):
                weights = gammaFull[j, :]
                total = np.sum(weights)

                self.mu[j] = np.sum(weights * velocities) / total

                var = np.sum(weights * (velocities - self.mu[j])**2) / total
                self.sigma[j] = np.sqrt(var)

        return
    
    def baum_forward(self, velocities,T,M):
        
        alpha = np.full((T, M), -np.inf)

        for s in range(M):
            alpha[0, s] = self.init[s] + self.emit_log_prob(velocities[0], s)

        for t in range(1, T):
            for j in range(M):
                terms = []
                for i in range(M):
                    terms.append(alpha[t-1, i] + self.trans[i, j])
                alpha[t, j] = self.log_sum_exp(np.array(terms)) + self.emit_log_prob(velocities[t], j)

        return alpha
    
    def baum_backward(self, velocities,T,M):
        
        beta = np.full((T, M), -np.inf)

        beta[T-1, :] = 0 

        for t in range(T-2, -1, -1):
            for i in range(M):
                terms = []
                for j in range(M):
                    terms.append(
                        self.trans[i, j] +
                        self.emit_log_prob(velocities[t+1], j) +
                        beta[t+1, j]
                    )
                beta[t, i] = self.log_sum_exp(np.array(terms))

        return beta
    
    '''
    code from "https://ristohinno.medium.com/baum-welch-algorithm-4d4514cf9dbe" still need to fully check it 
    # DONE #TODO: adapt for log probs
    def baum_welch(O, a, b, initial_distribution, n_iter=100):
        #http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/
        M = a.shape[0]
        T = len(O)
        for n in range(n_iter):
            ###estimation step
            alpha = forward(O, a, b, initial_distribution)
            beta = backward(O, a, b)
            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                # joint probab of observed data up to time t @ transition prob * 
                #emisssion prob at t+1 @ joint probab of observed data from at t+1
                denominator = (alpha[t, :].T @ a * b[:, O[t + 1]].T) @ beta[t + 1, :]
                for i in range(M):
                    numerator = alpha[t, i] * a[i, :] * b[:, O[t + 1]].T * beta[t + 1, :].T
                    xi[i, :, t] = numerator / denominator
            gamma = np.sum(xi, axis=1)
            ### maximization step
            a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            # Add additional T'th element in gamma
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
            K = b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                b[:, l] = np.sum(gamma[:, O == l], axis=1)
            b = np.divide(b, denominator.reshape((-1, 1)))
        return a, b


    '''
    
    def viterbi(self, velocities):

        # init step

        T = len(velocities)

        prob = np.full((T, self.states), -np.inf)
        prev = np.zeros((T, self.states), dtype=int)

        for s in range(self.states):
            prob[0, s] = self.init[s] + self.emit_log_prob(velocities[0], s)

        # main loop

        for t in range(1, T):
            for state1 in range(self.states):
                bestP = -np.inf
                bestState = 0
                for state2 in range(self.states):
                    newP = prob[t-1, state2] + self.trans[state2, state1] + self.emit_log_prob(velocities[t], state1)
                    if newP > bestP:
                        bestP = newP
                        bestState = state2
                prob[t, state1] = bestP
                prev[t, state1] = bestState

        
        # backtrack
        
        path = np.zeros(T, dtype=int)

        path[T-1] = np.argmax(prob[T-1])

        for t in range(T-2, -1, -1):
            path[t] = prev[t+1, path[t+1]]

        return path

    
    

'''
function Viterbi(states, init, trans, emit, obs) is
    input states: S hidden states
    input init: initial probabilities of each state
    input trans: S × S transition matrix
    input emit: S × O emission matrix
    input obs: sequence of T observations

    prob ← T × S matrix of zeroes
    prev ← empty T × S matrix
    for each state s in states do
        prob[0][s] = init[s] * emit[s][obs[0]]

    for t = 1 to T - 1 inclusive do // t = 0 has been dealt with already
        for each state s in states do
            for each state r in states do
                new_prob ← prob[t - 1][r] * trans[r][s] * emit[s][obs[t]]
                if new_prob > prob[t][s] then
                    prob[t][s] ← new_prob
                    prev[t][s] ← r

    path ← empty array of length T
    path[T - 1] ← the state s with maximum prob[T - 1][s]
    for t = T - 2 to 0 inclusive do
        path[t] ← prev[t + 1][path[t + 1]]

    return path
end
'''

@register_event_detection
def ihmm(
        positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        mu: list[int] | np.ndarray | None = None, 
        sigma: list[int] | np.ndarray | None = None, 
        init_state : list[int] | np.ndarray | None = None,
        transition_probabilities: list[list[float]] | np.ndarray | None = None,
        initialization: str | None = None,
        name: str = 'fixation',
) -> Events:
    
    positions = np.array(positions)

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


    # TODO: Implement other dimension checks for inputs





    # convert into velocities (1D velocities vector)

    # TODO: Optimize, maybe implement different vel algorithms/connect to pos2vel method/make use of the velocity column if present

    velocities = []

    for ind in range(len(positions)-1):

        i=ind+1
        x_i= positions[i-1][0]
        x_i_1 = positions[i][0]
        y_i = positions[i-1][1]
        y_i_1 = positions[i][1]
        t_i = timesteps[i-1]
        t_i_1 = timesteps[i]

        dt = t_i_1 - t_i

        if dt == 0:
            v_i = 0
        else:
            v_i = np.sqrt((x_i_1 - x_i)**2 + (y_i_1 - y_i)**2) / dt

        velocities.append(v_i)

    velocities = np.array(velocities)


    # Init 2 state HMM

    # DONE # TODO: find reasonable def paramenters
    defaults={
        "mu": [np.percentile(velocities, 30), np.percentile(velocities, 80)], #DATA BASED init  #[1.0, 10.0],
        "sigma": [np.var(velocities)/2, np.var(velocities)], # #DATA BASED init   #[1.0, 1.0],
        "init":[0.5, 0.5],  # dummy average values should be fine for long sequences
        "trans":[[0.95, 0.05],[0.05, 0.95]] # based on Salvucci's paper diagram
    }

    reestimate=False

    match initialization:
        case "reestimation":
            # TODO: Implement Baum-Welch
            reestimate = True
            _mu = defaults["mu"]
            _sigma=defaults["sigma"]
            _init = defaults["init"]
            _trans = defaults["trans"]

        case "default":
            _mu = defaults["mu"]
            _sigma=defaults["sigma"]
            _init = defaults["init"]
            _trans = defaults["trans"]
        case _:
            if mu:
                _mu=mu
            else:
                _mu = defaults["mu"]
            if sigma:
                _sigma = sigma
            else:
                _sigma= defaults["sigma"]
            if init_state:
                _init = init_state
            else:
                _init = defaults["init"]
            if transition_probabilities:
                _trans = transition_probabilities
            else:
                _trans = defaults["trans"]

    hmm = HMM(states= 2 ,mu=_mu,sigma=_sigma,initial_state=_init,transition_matrix=_trans)

    if reestimate:
        hmm.baum_welch(velocities=velocities)

    # inference the hmm 

    states = hmm.viterbi(velocities=velocities)

    # collapse states
    
    onsets_arr = []
    offsets_arr = []

    prevState = states[0]

    if prevState == 0:
        onsets_arr.append(0)

    for i, state in enumerate(states[1:], start=1):

        if state == 0:
            if prevState != 0:
                onsets_arr.append(i)
            #else:
            #    pass 
        else:
            #if prevState != 0:
            #    pass
            if prevState == 0:
                offsets_arr.append(i-1)

        prevState = state
    
    if prevState == 0:
        offsets_arr.append(len(states) - 1)

    onsets_arr = np.array(onsets_arr)
    offsets_arr = np.array(offsets_arr)
    

    # DONE # TODO: transform in event object

    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)

    return events

































    '''
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

    if dispersion_threshold <= 0:
        raise ValueError('dispersion_threshold must be greater than 0')
    if minimum_duration <= 0:
        raise ValueError('minimum_duration must be greater than 0')
    if not isinstance(minimum_duration, int):
        raise TypeError(
            'minimum_duration must be of type int'
            f' but is of type {type(minimum_duration)}',
        )

    onsets = []
    offsets = []

    # Infer minimum duration in number of samples.
    # This implementation is currently very restrictive.
    # It requires that the interval between timesteps is constant.
    # It requires that the minimum duration is divisible by the constant interval between timesteps.
    timesteps_diff = np.diff(timesteps)
    if not np.all(timesteps_diff == timesteps_diff[0]):
        raise ValueError('interval between timesteps must be constant')
    if not minimum_duration % timesteps_diff[0] == 0:
        raise ValueError(
            'minimum_duration must be divisible by the constant interval between timesteps',
        )
    if (minimum_sample_duration := int(minimum_duration // timesteps_diff[0])) < 2:
        raise ValueError('minimum_duration must be longer than the equivalent of 2 samples')

    # Initialize window over first points to cover the duration threshold
    win_start = 0
    win_end = minimum_sample_duration

    while win_start < len(timesteps) and win_end <= len(timesteps):

        # Initialize window over first points to cover the duration threshold.
        # This automatically extends the window to the specified minimum event duration.
        win_end = max(win_start + minimum_sample_duration, win_end)
        win_end = min(win_end, len(timesteps))
        if win_end - win_start < minimum_sample_duration:
            break

        if dispersion(positions[win_start:win_end]) <= dispersion_threshold:
            # Add additional points to the window until dispersion > threshold.
            while dispersion(positions[win_start:win_end]) < dispersion_threshold:
                # break if we reach end of input data
                if win_end == len(timesteps):
                    break

                win_end += 1

            # check for np.nan values
            if np.sum(np.isnan(positions[win_start:win_end - 1])) > 0:
                tmp_candidates = [np.arange(win_start, win_end - 1, 1)]
                tmp_candidates = filter_candidates_remove_nans(
                    candidates=tmp_candidates,
                    values=positions,
                )
                # split events if include_nan == False
                if not include_nan:
                    tmp_candidates = events_split_nans(
                        candidates=tmp_candidates,
                        values=positions,
                    )

                # Filter all candidates by minimum duration.
                tmp_candidates = [
                    candidate for candidate in tmp_candidates
                    if len(candidate) >= minimum_sample_duration
                ]
                for candidate in tmp_candidates:
                    onsets.append(timesteps[candidate[0]])
                    offsets.append(timesteps[candidate[-1]])

            else:
                # Note a fixation at the centroid of the window points.

                onsets.append(timesteps[win_start])
                offsets.append(timesteps[win_end - 1])

            # Remove window points from points.
            # Initialize new window excluding the previous window
            win_start = win_end
        else:
            # Remove first point from points.
            # Move window start one step further without modifying window end.
            win_start += 1

    # Create proper flat numpy arrays.
    onsets_arr = np.array(onsets).flatten()
    offsets_arr = np.array(offsets).flatten()

    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)
    return events
    '''
