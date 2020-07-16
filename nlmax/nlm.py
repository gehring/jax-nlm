from collections import abc
import itertools

import jax
import jax.numpy as jnp


EXPAND = "expand"
REDUCE = "reduce"


def nlm_mask(x):
    _mask = mask = 1 - jnp.eye(x.shape[0])
    for _ in range(x.ndim - 2):
        mask = jnp.tensordot(mask, _mask, 0)
    return mask


def nlm_reduce(x, mask=None):
    if mask is None:
        mask = nlm_mask(x)
    return jnp.concatenate([jnp.max(x * mask, axis=-2),
                            jnp.min(x * mask + (1 - mask), axis=-2)],
                           axis=-1)


def nlm_expand(x):
    expanded_shape = list(x.shape)
    expanded_shape.insert(-2, expanded_shape[-2])
    return jnp.broadcast_to(x[..., None, :], expanded_shape)


def nlm_expand_reduce(xs, expand_fn=None, reduce_fn=None, mode=None):
    if expand_fn is None:
        expand_fn = nlm_expand
    if reduce_fn is None:
        reduce_fn = nlm_reduce

    # Shift the reduced and expanded inputs and pad with None.
    if mode is None:
        expanded = [None] + [expand_fn(x) for x in xs[:-1]]
        reduced = [reduce_fn(x) for x in xs[1:]] + [None]
    elif mode == EXPAND:
        expanded = [None] + [expand_fn(x) for x in xs]
        reduced = [reduce_fn(x) for x in xs[1:]] + [None] * 2
        xs = xs + [None]
    elif mode == REDUCE:
        expanded = [None] + [expand_fn(x) for x in xs[:-2]]
        reduced = [reduce_fn(x) for x in xs[1:]]
        xs = xs[:-1]

    # Zip the reduced and expanded values with the inputs and filter out the
    # None padding.
    xs = [[v for v in x if v is not None] for x in zip(xs, reduced, expanded)]

    return [jnp.concatenate(x, axis=-1) for x in xs]


def nlm_permute(x):
    if x.ndims > 2:
        x = jnp.concatenate([
                jnp.transpose(x, list(perm) + [x.ndims - 1])
                for perm in itertools.permutations(range(x.ndims))
            ])
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
