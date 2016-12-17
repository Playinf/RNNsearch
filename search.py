# search.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy as np

from tensorflow.python.util import nest


# score: a beam_size * num_vars matrix, represent current score
# n: max number of elements to select
# threshold: prune if score < best + threshold
def find_nbest(score, n, threshold=None):
    num_vars = score.shape[1]

    score = score.flatten()
    nbest = np.argpartition(score, n)[:n]

    beam_indices = nbest / num_vars
    var_indices = nbest % num_vars
    nbest_score = score[nbest]

    if threshold:
        best = np.max(nbest_score)
        cond = nbest_score > best + threshold
        nbest_score = nbest_score[cond]
        beam_indices = beam_indices[cond]
        var_indices = var_indices[cond]

    return nbest_score, beam_indices, var_indices


# nested: a nested structure of shape batch * dim
# indices: indices to select
def select_nbest(nested, indices):
    if not isinstance(nested, (list, tuple)):
        return nested[indices]

    flat_list = nest.flatten(nested)
    selected_list = [item[indices] for item in flat_list]

    return nest.pack_sequence_as(nested, selected_list)


class beam:

    def __init__(self, beamsize, threshold=None):
        self.size = beamsize
        self.threshold = threshold
        self.score = []
        self.candidate = []

    def prune(self, dist, cond, prev_beam):
        prev_score = np.array(prev_beam.score, dist.dtype)
        score = prev_score[:, None] - dist

        outputs = find_nbest(score, self.size, self.threshold)
        nbest_score, beam_indices, var_indices = outputs

        finished = []
        remained = []

        for i, (bid, vid) in enumerate(zip(beam_indices, var_indices)):
            prev_candidate = prev_beam.candidate
            candidate = prev_candidate[bid] + [vid]

            if cond(candidate):
                finished.append([candidate, nbest_score[i]])
            else:
                remained.append(i)
                self.candidate.append(candidate)
                self.score.append(nbest_score[i])

        return finished, beam_indices[remained], var_indices[remained]
