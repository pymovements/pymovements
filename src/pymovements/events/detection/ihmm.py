from __future__ import annotations

import numpy as np

from pymovements._utils import _checks
from pymovements.events._utils._filters import events_split_nans
from pymovements.events._utils._filters import filter_candidates_remove_nans
from pymovements.events.detection.library import register_event_detection
from pymovements.events.events import Events

'''
def dispersion(positions: list[list[float]] | np.ndarray) -> float:
    """Compute the dispersion of a group of consecutive points in a 2D position time series.

    The dispersion is defined as the sum of the differences between
    the points' maximum and minimum x and y values

    Parameters
    ----------
    positions: list[list[float]] | np.ndarray
        Continuous 2D position time series.

    Returns
    -------
    float
        Dispersion of the group of points.
    """
    return sum(np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0))'''

class HMM:

    # TODO: some string representation method? To inspect states and so on

    def __init__(self,states,mu,sigma,initial_state,transition_matrix):

        # TODO: implement different initializations outside/inside

        self.states = states

        self.init = np.log(initial_state)

        self.mu=mu

        self.sigma = sigma

        self.trans = np.log(transition_matrix)

        self.emit = [] # TODO: initialize properly
        
        return
    
    def viterbi(self, velocities):

        # init step

        prob = np.zeros(shape=(len(velocities),self.states)) 
        prev = []

        for s in range(self.states):
            prob[0][s] = self.initial_state[s] * self.emit[s][velocities[0]] # TODO: check indeces for velocities[0]

        # main loop

        

        return
    

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
        minimum_duration: int = 100,
        dispersion_threshold: float = 1.0,
        include_nan: bool = False,
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
    hmm = HMM(states=2,mu=[],sigma=[],initial_state=[],transition_matrix=[])

    # inference the hmm 

    states = hmm.viterbi(velocities=velocities)

    # collapse states

    

    return 































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
