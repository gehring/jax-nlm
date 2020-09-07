import collections
import itertools

import gin

import numpy as np

import jax
import jax.scipy
import jax.numpy as jnp
from jax.experimental import optix

import haiku as hk

import nlmax

from difflogic.dataset.graph import dataset

TrainingData = collections.namedtuple("TrainingData", ["predicates", "targets"])
OptimizationState = collections.namedtuple("OptimizationState",
                                           ["params", "opt_state"])

gin.external_configurable(jax.scipy.special.logsumexp)
gin.parse_config(
    """
    # nlm_reduce.reduce_fn = @logsumexp
    """
)


def preprocess_batch(batch):
    predicates, targets = zip(*[(d["relations"], d["target"]) for d in batch])
    return TrainingData([np.stack(predicates)], np.stack(targets))


def collect_batch(data_generator, n):
    "Collect data into fixed-length chunks or blocks"
    # Taken from the itertools recipe for the `grouper` function.
    args = [iter(data_generator)] * n
    return itertools.zip_longest(*args)


def make_network(num_nlm_layers, num_hidden_units, output_dim, max_arity=3):

    @hk.without_apply_rng
    @hk.transform
    def model(inputs):
        assert len(inputs) < num_nlm_layers
        outputs = inputs
        for i in range(num_nlm_layers - 1):
            mode = None
            current_arity = max([x.ndim - 2 for x in outputs])
            if current_arity < max_arity:
                mode = "expand"

            outputs = nlmax.nlm_layer(
                outputs,
                lambda x: jax.nn.sigmoid(hk.Linear(num_hidden_units)(x)),
                mode=mode,
            )

        outputs = nlmax.nlm_layer(
            outputs,
            lambda x: hk.Linear(output_dim)(x),
        )

        return outputs[2]

    return model


def cross_entropy(labels, logits):
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(labels, num_classes)
    return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)


def train(batch_size, max_iter, num_test_samples):
    model = make_network(4, 8, output_dim=2)
    optimizer = optix.adam(0.005)

    data_generator = collect_batch(
        dataset.FamilyTreeDataset(
            task="grandparents",
            epoch_size=10,
            nmin=20),
        n=batch_size,
    )

    @jax.jit
    def evaluate_batch(params, data):
        logits = model.apply(params, data.predicates)
        correct = jnp.all(data.targets == jnp.argmax(logits, axis=-1), axis=-1)
        ce_loss = jnp.sum(cross_entropy(data.targets, logits), axis=-1)
        return correct, ce_loss

    def evaluate(params):
        eval_data_gen = collect_batch(
            dataset.FamilyTreeDataset(
                task="grandparents",
                epoch_size=10,
                nmin=50),
            n=batch_size,
        )
        correct, ce_loss = zip(*[
            evaluate_batch(params, preprocess_batch(data))
            for data in itertools.islice(eval_data_gen, num_test_samples)
        ])
        correct = np.concatenate(correct, axis=0)
        ce_loss = np.concatenate(ce_loss, axis=0)

        return np.mean(correct), np.mean(ce_loss)

    def loss(params, data):
        logits = model.apply(params, data.predicates)
        return jnp.mean(cross_entropy(data.targets, logits))

    @jax.jit
    def update(data, params, opt_state):
        batch_loss, dparam = jax.value_and_grad(loss)(params, data)

        updates, opt_state = optimizer.update(dparam, opt_state)
        params = optix.apply_updates(params, updates)
        return batch_loss, OptimizationState(params, opt_state)

    rng = jax.random.PRNGKey(0)
    init_params = model.init(
        rng,
        preprocess_batch(next(data_generator)).predicates,
    )
    state = OptimizationState(init_params, optimizer.init(init_params))

    for i, batch in enumerate(data_generator):
        if i >= max_iter:
            break

        batch = preprocess_batch(batch)
        batch_loss, state = update(batch, *state)

        if (i + 1) % 100 == 0:
            print(i + 1, evaluate(state.params))


train(4, 10000, 20)
