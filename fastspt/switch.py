from fastspt.core import Track
import numpy as np


def get_switching_matrix(
    tracks: [Track], column="states", n_states=2, return_raw_counts=False
):
    """
    Computes switching matrix using state integers in the `column`
    """
    n = n_states
    matrix = np.zeros((n, n))

    def get_matrix_one_track(track):
        states = track.col(column).astype("int").flat
        for curr_state, next_state in zip(states[:-1], states[1:]):
            try:
                matrix[curr_state, next_state] += 1
            except IndexError:
                raise ValueError(
                    f"Unexpected value occured {[curr_state, next_state]} in the track \n{track}, \nexpected values: {tuple(np.arange(n_states))}"
                )

    _ = list(map(get_matrix_one_track, tracks))
    norm = matrix.sum(axis=1).reshape(n, 1)
    norm_matrix = matrix / norm
    if return_raw_counts:
        return (norm_matrix, matrix)
    return norm_matrix
