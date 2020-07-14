from collections import abc
import itertools

import jax
import jax.numpy as jnp


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


def nlm_expand_reduce(xs, expand_fn=None, reduce_fn=None):
    if expand_fn is None:
        expand_fn = nlm_expand
    if reduce_fn is None:
        reduce_fn = nlm_reduce

    expanded = [expand_fn(x) for x in xs[:-1]]
    reduced = [reduce_fn(x) for x in xs[1:]]

    xs = [(xs[0], reduced[0])] + list(zip(xs[1:], reduced, expanded))
    return [jnp.concatenate(x, axis=-1) for x in xs]


def nlm_permute(x):
    if x.ndims > 2:
        x = jnp.concatenate([
                jnp.transpose(x, list(perm) + [x.ndims - 1])
                for perm in itertools.permutations(range(x.ndims))
            ])
    return x


def nlm_layer(xs, mlp_cls, residual=False):
    # If `mlp_cls` is not iterable we assume there is a single, shared
    # constructor for all MLPs.
    if isinstance(mlp_cls, abc.Iterable):
        mlp_cls = itertools.repeat(mlp_cls)

    # add a batch dimension to the "preprocessing" step
    @jax.vmap
    def _nlm_preprocess(inputs):
        outputs = nlm_expand_reduce(inputs)
        return [nlm_permute(x) for x in outputs]

    outputs = [mlp()(x) for mlp, x in zip(mlp_cls, _nlm_preprocess(xs))]
    if residual:
        outputs = [jnp.concatenate(x, -1) for x in zip(outputs, xs)]

    return outputs
