import os, time
from typing import Any
from functools import partial

from tqdm import tqdm
import optax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state, orbax_utils
import tensorboardX as tbx


key = jax.random.PRNGKey(0)


def banner_message(message):
    if isinstance(message, str):
        message = [message]
    elif not isinstance(message, list) or not all(isinstance(m, str) for m in message):
        raise ValueError("message should be a string or a list of strings.")

    msg_len = max(46, max(len(msg) for msg in message))

    # Top border
    print("\33[1;32m╔═{:═<{width}}═╗".format("", width=msg_len))
    # Message lines
    for msg in message:
        print("║ {:^{width}} ║".format(msg, width=msg_len))
    # Bottom border
    print("╚═{:═<{width}}═╝\33[0m".format("", width=msg_len))


def lr_schedule(lr, steps_per_epoch, epochs=100, warmup=5):
    return optax.warmup_cosine_decay_schedule(
        init_value=lr / 10,
        peak_value=lr,
        end_value=1e-5,
        warmup_steps=steps_per_epoch * warmup,
        decay_steps=steps_per_epoch * (epochs - warmup),
    )


# implement TrainState
class TrainState(train_state.TrainState):
    batch_stats: Any


@jax.jit
def loss_fn(logits, labels):
    loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, 10)).mean()
    loss_dict = {'loss': loss}
    return loss, loss_dict


@jax.jit
def eval_step(state: TrainState, batch):
    x, y = batch
    logits = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats,
        }, x, train=False)
    acc = jnp.equal(jnp.argmax(logits, -1), y).mean()
    return acc


@partial(jax.jit, static_argnums=(3,))
def train_step(state: TrainState, batch, opt_state, loss_fn):
    x, y = batch
    def compute_loss(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x,
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout': key}
        )
        loss, loss_dict = loss_fn(logits, y)
        return loss, (loss_dict, updates)

    (_, (loss_dict, updates)), grads = jax.value_and_grad(compute_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=updates['batch_stats'])
    _, opt_state = state.tx.update(grads, opt_state)
    return state, loss_dict, opt_state


def load_ckpt(state, ckpt_dir, step=None):
    if ckpt_dir is None or not os.path.exists(ckpt_dir):
        banner_message(["No checkpoint was loaded", "Training from scratch"])
        return state

    ckpt_dir = os.path.abspath(ckpt_dir)
    step = step or max(int(f) for f in os.listdir(ckpt_dir) if f.isdigit())

    ckpt_path = os.path.join(ckpt_dir, str(step), "default")
    manager = ocp.PyTreeCheckpointer()
    return manager.restore(ckpt_path, item=state)


def fit(state,
        train_ds, test_ds,
        num_epochs=100,
        loss_fn=loss_fn,
        eval_step=None,
        eval_freq=1,
        log_name='default',
        hparams=None,
    ):
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
    # checkpoint manager see https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#with-orbax
    options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=1,
        save_interval_steps=eval_freq,
    )
    manager = ocp.CheckpointManager(ckpt_path, options=options)
    # logging
    banner_message(["Start training", "Device > {}".format(", ".join([str(i) for i in jax.devices()]))])
    timestamp = time.strftime("%m%d%H%M%S")
    writer = tbx.SummaryWriter("logs/{}_{}".format(log_name, timestamp))
    opt_state = state.tx.init(state.params)
    best_acc = .0
    # start training
    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(train_ds)
        for batch in pbar:
            state, loss_dict, opt_state = train_step(state, batch, opt_state, loss_fn)
            lr = opt_state.hyperparams['learning_rate']
            pbar.set_description(f'Epoch {epoch:3d}, lr: {lr:.7f}, loss: {loss_dict["loss"]:.4f}')

            if state.step % 10 == 0 or state.step == 1:
                writer.add_scalar('train/learning_rate', lr, state.step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v, state.step)

            writer.flush()

        if eval_step is None:
            manager.save(epoch, args=ocp.args.StandardSave(state))

        elif epoch % eval_freq == 0:
            acc = []
            for batch in test_ds:
                a = eval_step(state, batch)
                acc.append(a)
            acc = jnp.stack(acc).mean()
            pbar.write(f'Epoch {epoch:3d}, test acc: {acc:.4f}')
            writer.add_scalar('test/accuracy', acc, epoch)

            if acc > best_acc:
                manager.save(epoch, args=ocp.args.StandardSave(state), metrics={'accuracy': acc})
                best_acc = acc

    manager.wait_until_finished()
    banner_message(["Training finished", f"Best test acc: {best_acc:.6f}"])
    if hparams is not None:
        writer.add_hparams(hparams, {'metric/accuracy': best_acc}, name='hparam')
    writer.close()
