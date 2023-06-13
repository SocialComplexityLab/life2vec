import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# MULTI HEAD SPARSE BLOCK ATTENTION UTILS


def torch_bmm_nd(inp_1, inp_2, ndim=None):
    """ Fast nd matrix multiplication """ ""
    return torch.bmm(
        inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])
    ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1]))


def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
    """ Fast nd matrix multiplication with transpose """ ""
    return torch.bmm(
        inp_1.reshape((-1,) + inp_1.shape[-2:]),
        inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2),
    ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))


def get_random_mask_with_heads(
    sequence_length: int,
    num_heads: int,
    blocked_mask,
    indexes,
    block_size: int,
    nr: int,
    batch_size: int,
):
    bm = (
        blocked_mask.unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(1)
        .expand(-1, num_heads, -1, -1, 1, nr)
        .transpose(-1, -3)
        .view(batch_size, num_heads, sequence_length // block_size, nr, block_size)
    )

    random_mask = (
        indexes.unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, num_heads, -1, -1, block_size)
    )
    # print(random_mask.shape, bm.shape)
    random_mask = torch.gather(bm, 2, random_mask)
    return torch.einsum(
        "blq,bhlk->bhlqk",
        blocked_mask[:, 1:-1],
        random_mask.view(
            batch_size, num_heads, sequence_length // block_size - 2, block_size * nr
        ),
    )

def blockify_with_heads(
    x, batch_size: int, num_heads: int, sequence_length: int, block_size: int
):
    """
    block_length: sequence_length // block_size
    """
    return x.reshape(
        batch_size, num_heads, sequence_length // block_size, block_size, -1
    )

def get_random_attention_indexes_with_heads(
    sequence_length: int,
    num_heads: int,
    block_size: int,
    num_random: int,
    num_neighbours: int = 3,
    margin: int = 1,
):
    """
    Selects random IDs for each row
    Output: Tensor, with IDs
    """
    assert num_neighbours == 3, "Not Implemented"
    bl = sequence_length // block_size
    di = np.diag_indices(bl, ndim=1)

    illegal_indices = np.concatenate(
        [
            np.zeros([1, bl]),
            di,
            np.roll(di, shift=-1),
            np.roll(di, shift=1),
            np.full([1, bl], bl - 1),
        ]
    ).transpose(-1, -2)

    def h(x, rn):
        return np.random.choice(
            [i for i in range(0, sequence_length // block_size) if i not in x],
            rn,
            replace=False,
        )

    res = []
    for _ in range(num_heads):
        res.append(
            np.apply_along_axis(h, 1, illegal_indices, rn=num_random)[margin:-margin]
        )

    return torch.LongTensor(np.stack(res, 0))


def get_gathered_indexes_with_heads(
    indexes,
    num_heads: int,
    batch_size: int,
    sequence_length: int,
    block_size: int,
    rn: int,
    hidden_size: int,
):
    """
    Map for random blocks
    Args:
        indexes: Tensor, with block_ids
    """
    assert indexes.shape[0] == num_heads, "Wrong number of heads"
    head_indexes = []
    for h in range(num_heads):
        head_indexes.append(
            get_gathered_indexes(
                indexes[h],
                batch_size=batch_size,
                sequence_length=sequence_length,
                block_size=block_size,
                hidden_size=hidden_size // num_heads,
                rn=rn,
            )
        )

    return torch.stack(head_indexes, 1)


# SINGLE HEAD SPARSE BLOCK ATTENTION UTILS


def get_random_attention_indexes(
    sequence_length: int,
    block_size: int,
    num_random: int,
    num_neighbours: int = 3,
    margin: int = 1,
):
    """
    Selects random IDs for each row
    Output: Tensor, with IDs
    """
    assert num_random < 3 and num_random >= 1, "Not Implemented"
    assert num_neighbours == 3, "Not Implemented"
    assert (
        sequence_length // block_size - 1 - num_neighbours > num_random
    ), "Number of random blocks is too large"

    bl = sequence_length // block_size
    di = np.diag_indices(bl, ndim=1)
    illegal_indices = np.concatenate(
        [
            np.zeros([1, bl]),
            di,
            np.roll(di, shift=-1),
            np.roll(di, shift=1),
            np.full([1, bl], bl - 1),
        ]
    ).transpose(-1, -2)

    def h(x, rn):
        return np.random.choice(
            [i for i in range(0, sequence_length // block_size) if i not in x],
            rn,
            replace=False,
        )

    # , illegal_indices[margin:-margin]
    return torch.LongTensor(
        np.apply_along_axis(h, 1, illegal_indices, rn=num_random)[margin:-margin]
    )


@torch.jit.script
def get_gathered_indexes(
    indexes,
    batch_size: int,
    sequence_length: int,
    block_size: int,
    rn: int,
    hidden_size: int,
):
    """
    Map for random blocks
    Args:
        indexes: Tensor, with block_ids
    """
    return (
        indexes.unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, -1, rn, block_size)
        .reshape(batch_size, -1, block_size * rn)
        .unsqueeze(-1)
        .expand(-1, -1, -1, hidden_size)
        .reshape(batch_size, -1, block_size, hidden_size)
    )


@torch.jit.script
def get_random_mask(
    sequence_length: int,
    blocked_mask,
    indexes,
    block_size: int,
    nr: int,
    batch_size: int,
):
    bm = (
        blocked_mask.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, -1, 1, nr)
        .transpose(-1, -3)
        .view(batch_size, sequence_length // block_size, nr, block_size)
    )
    random_mask = (
        indexes.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, block_size)
    )
    random_mask = torch.gather(bm, 1, random_mask)
    return torch.einsum(
        "blq,blk->blqk",
        blocked_mask[:, 1:-1],
        random_mask.view(
            batch_size, sequence_length // block_size - 2, block_size * nr
        ),
    )


@torch.jit.script
def get_padding_mask(x, padding_token: int = 0):
    return (x == padding_token).long()


@torch.jit.script
def get_padding2attention_mask(padding_mask):
    mask = torch.einsum("bf,bt->bft", padding_mask, padding_mask)
    # Create 2D attention mask
    return torch.unsqueeze(mask, 1)


@torch.jit.script
def get_band_mask(blocked_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
        blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].
    Returns:
        float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                              block_size,  3*to_block_size].
    """

    exp_blocked_to_pad = torch.cat(
        [blocked_mask[:, 1:-3], blocked_mask[:, 2:-2], blocked_mask[:, 3:-1]], 2
    )
    band_mask = torch.einsum(
        "blq,blk->blqk", blocked_mask[:, 2:-2], exp_blocked_to_pad
    ).unsqueeze(1)
    return band_mask


@torch.jit.script
def blockify(x, batch_size: int, block_length: int, block_size: int):
    """
    block_length: sequence_length // block_size
    """
    return x.reshape(batch_size, block_length, block_size, -1)


def simulate_sparse_mask(sequence_length, block_size, rand_attn):
    r = np.kron(
        np.diag(np.ones(sequence_length // block_size)),
        np.ones((block_size, block_size), dtype="int"),
    )
    r = torch.LongTensor(r)
    r = torch.max(r, torch.roll(r, block_size, 1))
    r = torch.max(r, torch.roll(r, -block_size, 1))
    r[:, :block_size] = 1
    r[:, -block_size:] = 1
    r[:block_size, :] = 1
    r[-block_size:, :] = 1
    ind = torch.LongTensor(range(1, sequence_length // block_size - 1))
    r = (
        torch.LongTensor(r)
        .view(
            sequence_length // block_size,
            block_size,
            sequence_length // block_size,
            block_size,
        )
        .permute(0, 2, 1, -1)
        .transpose(-1, -2)
    )
    for i in range(rand_attn.shape[1]):
        r[ind, rand_attn[:, i]] = 1

    return (
        r.transpose(-1, -2).permute(0, 2, 1, 3).view(sequence_length, sequence_length)
    )
