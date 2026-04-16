from __future__ import annotations

import numpy as np

from pymovements._utils import _checks
from pymovements.events._utils._filters import events_split_nans
from pymovements.events._utils._filters import filter_candidates_remove_nans
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events


class HMM:

    # TODO: some string representation method? To inspect states and so on

    def __init__(
            self,
            states : int,
            mu: list[float] | np.ndarray,
            sigma: list[float] | np.ndarray,
            initial_state: list[float] | np.ndarray,
            transition_matrix: list[list[float]] | np.ndarray):

        self.states = states

        self.init = np.log(initial_state)

        self.mu=mu

        self.sigma = sigma

        self.trans = np.log(transition_matrix)
        
        return
    
    def emit_log_prob(self, v, s):
        #v = float(v)
        #print(f"v is : {v}")
        mu = self.mu[s]
        sigma = self.sigma[s]

        sigma = max(sigma, 1e-6)  

        #print("emit")

        #print(-0.5 * np.log(2*np.pi*sigma**2) - ((v - mu)**2) / (2*sigma**2))

        return -0.5 * np.log(2*np.pi*sigma**2) - ((v - mu)**2) / (2*sigma**2)
    
    def log_sum_exp(
            self,
            arr: np.ndarray) -> float:
        m = np.max(arr)
        #print("logsum2")
        #print(m + np.log(np.sum(np.exp(arr - m))))
        #if np.all(np.isneginf(arr)):
            #print(-np.inf)
        #    return -np.inf
        return m + np.log(np.sum(np.exp(arr - m)))

    def baum_welch(
            self,
            velocities: list[float] | np.ndarray,
            max_iters: int,
            epsilon: float = 1e-4) -> dict[str, np.ndarray]:

        T = len(velocities)
        M = self.states

        # DONE # TODO: Implement convergence instead of iters

        prev_log_likelihood = -np.inf

        

        for _ in range(max_iters):

            print(prev_log_likelihood)

            alpha = self.baum_forward(velocities,T,M)

            beta = self.baum_backward(velocities,T,M)

            xi = np.zeros((M, M, T-1))

            for t in range(T-1):
                denom_terms = []

                for i in range(M):
                    for j in range(M):
                        denom_terms.append(
                            alpha[t, i] +
                            self.trans[i, j] +
                            self.emit_log_prob(velocities[t+1], j) +
                            beta[t+1, j]
                        )

                denom = self.log_sum_exp(np.array(denom_terms))

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

            gamma_full = np.zeros((M, T))
            gamma_full[:, :-1] = gamma

            last = alpha[T-1] + beta[T-1]
            last = np.exp(last - self.log_sum_exp(last))
            gamma_full[:, -1] = last

            #e = 1e-12
            #print(f"e:{e}")

            self.init = np.log(gamma_full[:, 0])
            #
            #self.init = np.log(gamma_full[:, 0] + e)

            for i in range(M):
                denom = np.sum(gamma_full[i, :-1])
                for j in range(M):
                    numer = np.sum(xi[i, j, :])
                    self.trans[i, j] = np.log(numer / denom)
                    #self.trans[i, j] = np.log((numer + e) / (denom + e))

        
            for j in range(M):
                weights = gamma_full[j, :]
                total = np.sum(weights)

                self.mu[j] = np.sum(weights * velocities) / total

                var = np.sum(weights * (velocities - self.mu[j])**2) / total
                self.sigma[j] = np.sqrt(var)

                #var = np.sum(weights * (velocities - self.mu[j])**2) / total
                #self.sigma[j] = np.sqrt(max(var, 1e-6))
                
            

            alpha_updated = self.baum_forward(velocities, T, M)

            log_likelihood = self.log_sum_exp(alpha_updated[-1])

            if abs(log_likelihood - prev_log_likelihood) < epsilon:
                break

            prev_log_likelihood = log_likelihood
            #if np.isnan(self.trans).any():
            #    print("NaN in trans")

            #if np.isnan(self.sigma).any():
            #    print("NaN in sigma")

            #if np.isnan(alpha).any():
            #    print("NaN in alpha")
            
            #print(self.trans)
            #print(self.sigma)
            #print(self.mu)

        return  {"mu":self.mu, "sigma":self.sigma, "init":self.init, "trans":self.trans}
    
    def baum_forward(
            self, 
            velocities: list[float] | np.ndarray,
            T: int,
            M: int) -> np.ndarray:
        
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
    
    def baum_backward(
            self, 
            velocities: list[float] | np.ndarray,
            T: int,
            M: int) -> np.ndarray:
        
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
    
    def viterbi(
            self,
            velocities: list[float] | np.ndarray) -> np.ndarray:

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
                    new_prob = prob[t-1, state2] + self.trans[state2, state1] + self.emit_log_prob(velocities[t], state1)
                    if new_prob > best_prob:
                        best_prob = new_prob
                        best_state = state2
                prob[t, state1] = best_prob
                prev[t, state1] = best_state

        # backtrack
        
        path = np.zeros(T, dtype=int)

        path[T-1] = np.argmax(prob[T-1])

        for t in range(T-2, -1, -1):
            path[t] = prev[t+1, path[t+1]]

        return path

    

@register_event_detection
def ihmm(
        positions: list[list[float]] | list[tuple[float, float]] | np.ndarray,
        timesteps: list[int] | np.ndarray | None = None,
        mu: list[float] | np.ndarray | None = None, 
        sigma: list[float] | np.ndarray | None = None, 
        init_state : list[float] | np.ndarray | None = None,
        transition_probabilities: list[list[float]] | np.ndarray | None = None,
        reestimation_max_iters: int = 100,
        initialization: str | None = None,
        name: str = 'fixation',
) -> Events:
    """
    documentation...
    """
    
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

    velocities = np.nan_to_num(velocities, nan=0.0) # maybe should be average?

    # Init 2 state HMM

    print(velocities)

    #if np.isnan(velocities).any():
    #    print("NaNs in velocities!")

    defaults={
        "mu": [np.percentile(velocities, 30), np.percentile(velocities, 80)], #DATA BASED init  #[1.0, 10.0],
        "sigma": [np.sqrt(np.var(velocities)/2), np.sqrt(np.var(velocities))], #[np.var(velocities)/2, np.var(velocities)], # #DATA BASED init   #[1.0, 1.0],
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
        optimal = hmm.baum_welch(velocities=velocities,max_iters=reestimation_max_iters)
        # print(optimal)

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
                offsets_arr.append(i-1)

        prev_state = state
    
    if prev_state == 0:
        offsets_arr.append(len(states) - 1)

    onsets_arr = np.array(onsets_arr)
    offsets_arr = np.array(offsets_arr)
    
    events = Events(name=name, onsets=onsets_arr, offsets=offsets_arr)

    return events



