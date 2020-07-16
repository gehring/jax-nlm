from collections import abc
import itertools

import jax
import jax.numpy as jnp

import numpy as np

EXPAND = "expand"
REDUCE = "reduce"


def nlm_mask(x):
    mask = 1.
    if x.ndim > 2:
        mask = 1 - np.eye(x.shape[0])
        if x.ndim > 3:
            masks = []
            for axes in itertools.combinations(range(x.ndim - 1), x.ndim - 3):
                masks.append(np.expand_dims(mask, axes))
            mask = np.product(masks)

    return jnp.expand_dims(mask, -1)


def nlm_reduce(x, mask=None):
    if x is None:
        return None
    if mask is None:
        mask = nlm_mask(x)
    return jnp.concatenate([jnp.max(x * mask, axis=-2),
                            jnp.min(x * mask + (1 - mask), axis=-2)],
                           axis=-1)


def nlm_expand(x, m):
    if x is None:
        return None
    expanded_shape = list(x.shape)
    expanded_shape.insert(-2, m)
    return jnp.broadcast_to(x[..., None, :], expanded_shape)


def pad_missing_arity(xs):
    max_n = max([x.ndim - 1 for x in xs])
    padded_xs = [None] * (max_n + 1)
    for x in xs:
        padded_xs[x.ndim - 1] = x

    return padded_xs


def nlm_expand_reduce(xs, expand_fn=None, reduce_fn=None, mode=None):
    if expand_fn is None:
        expand_fn = nlm_expand
    if reduce_fn is None:
        reduce_fn = nlm_reduce

    # We assume `xs` is sorted by N-arity and second to last axis has dimension
    # equal to the number of objects.
    m = xs[-1].shape[-2]

    xs = pad_missing_arity(xs)

    # Shift the reduced and expanded inputs and pad with None.
    if mode is None:
        expanded = [None] + [expand_fn(x, m) for x in xs[:-1]]
        reduced = [reduce_fn(x) for x in xs[1:]] + [None]
    elif mode == EXPAND:
        expanded = [None] + [expand_fn(x, m) for x in xs]
        reduced = [reduce_fn(x) for x in xs[1:]] + [None] * 2
        xs = xs + [None]
    elif mode == REDUCE:
        expanded = [None] + [expand_fn(x, m) for x in xs[:-2]]
        reduced = [reduce_fn(x) for x in xs[1:]]
        xs = xs[:-1]

    # Zip the reduced and expanded values with the inputs and filter out the
    # None padding.
    xs = [[v for v in x if v is not None] for x in zip(xs, reduced, expanded)]

    return [jnp.concatenate(x, axis=-1) for x in xs]


def nlm_permute(x):
    if x.ndim > 2:
        x = jnp.concatenate(
            [
                jnp.transpose(x, list(perm) + [x.ndim - 1])
                for perm in itertools.permutations(range(x.ndim - 1))
            ],
            axis=-1,
        )
    return x


def nlm_layer(xs, mlps, residual=False, mode=None):
    # If `mlps` is not iterable we assume there is a single, shared callable.
    if not isinstance(mlps, abc.Iterable):
        mlps = itertools.repeat(mlps)

    # add a batch dimension to the "preprocessing" step
    @jax.vmap
    def _nlm_preprocess(inputs):
        outputs = nlm_expand_reduce(inputs, mode=mode)
        return [nlm_permute(x) for x in outputs]

    outputs = [mlp(x) for mlp, x in zip(mlps, _nlm_preprocess(xs))]
    if residual:
        outputs = [jnp.concatenate(x, -1) for x in zip(outputs, xs)]

    return outputs
