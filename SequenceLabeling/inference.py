# pylint: disable=C0103
"""inference"""

import mindspore
from mindspore import ops

class ViterbiDecoder:
    """
    Viterbi Decoder.
    """

    def __init__(self, tag_map):
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def decode(self, scores, lengths):
        """
        :param scores: CRF scores
        :param lengths: word sequence lengths
        :return: decoded sequences
        """
        batch_size = scores.shape[0]

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = ops.zeros((batch_size, self.tagset_size))

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = ops.ones((batch_size, int(max(lengths)), self.tagset_size)) * self.end_tag

        for t in range(max(lengths)):
            batch_size_t = sum(1 for l in lengths if l > t)  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
                backpointers[:batch_size_t, t, :] = ops.ones((int(batch_size_t), self.tagset_size),
                                                               dtype=mindspore.int32) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t], backpointers[:batch_size_t, t, :] = ops.max(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    axis=1)  # (batch_size, tagset_size)

        # Decode/trace best path backwards
        decoded = ops.zeros((batch_size, backpointers.shape[1]), dtype=mindspore.int32)
        pointer = ops.ones((batch_size, 1),
                             dtype=mindspore.int32) * self.end_tag  # the pointers at the ends are all <end> tags

        for t in list(reversed(range(backpointers.shape[1]))):
            decoded[:, t] = ops.gather_elements(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert all(ops.equal(decoded[:, 0], ops.ones((batch_size), dtype=mindspore.int32) * self.start_tag))

        # Remove the <starts> at the beginning, and append with <ends> (to compare to targets, if any)
        decoded = ops.cat([decoded[:, 1:], ops.ones((batch_size, 1), dtype=mindspore.int32) * self.start_tag],
                            axis=1)

        return decoded
