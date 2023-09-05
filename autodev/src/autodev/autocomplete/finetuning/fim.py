"""
The code in this module is based on https://github.com/loubnabnl/santacoder-finetuning
"""
import numpy as np

from autodev.autocomplete.fim_config import FIMTokenIds


def permute(
    sample,
    np_rng,
    fim_token_ids: FIMTokenIds,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), fim_token_ids.pad_token_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    [fim_token_ids.prefix_token_id, fim_token_ids.suffix_token_id],
                    suffix,
                    [fim_token_ids.middle_token_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [fim_token_ids.prefix_token_id],
                    prefix,
                    [fim_token_ids.suffix_token_id],
                    suffix,
                    [fim_token_ids.middle_token_id],
                    middle,
                ]
            )
    else:
        # don't do FIM preproc
        new_sample = sample

    return list(new_sample), np_rng
