import os, sys, time, glob, random

import yaml
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from fit import *
from model.loss import *
from model.unet import *
from model.dataloader import SynthDataLoader


key = jax.random.PRNGKey(0)
cfg = yaml.safe_load(open("config.yaml"))
print(cfg)

train_dl = SynthDataLoader(cfg["train_dir"], cfg["img_size"], cfg["batch_size"], shuffle=True)
test_dl = SynthDataLoader(cfg["test_dir"], cfg["img_size"], cfg["batch_size"], shuffle=False)
lr_fn = lr_schedule(cfg["lr"], len(train_dl), cfg["epochs"], cfg["warmup"])


@jax.jit
def loss_fn(pred, target):
    pred_mask, pred_hmap = pred
    y_mask, y_hmap = target
    loss_mask = tversky_loss(pred_mask, y_mask)
    loss_hmap = focal_loss(pred_hmap, y_hmap)
    loss = loss_mask + loss_hmap

    loss_dict = {
        'loss': loss,
        'loss_mask': loss_mask,
        'loss_hmap': loss_hmap,
    }
    return loss, loss_dict


@jax.jit
def eval_step(state: TrainState, batch):
    imgs, (y_mask, y_hmap) = batch
    (pred_mask, pred_hmap) = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
    }, imgs, train=False)
    acc = mIoU_acc(pred_mask, y_mask)
    return acc


if __name__ == "__main__":
    x = jnp.ones((1, 64, 64, 1))

    model = UNetV3(features=cfg["features"], max_mask=cfg["max_mask"])
    var = model.init(key, x)
    params = var['params']
    batch_stats = var['batch_stats']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.inject_hyperparams(optax.nadam)(lr_fn),
    )

    # state = load_ckpt(state, cfg["ckpt"])

    fit(state, train_dl, test_dl,
        loss_fn=loss_fn,
        eval_step=eval_step,
        num_epochs=cfg["epochs"],
        eval_freq=10,
        log_name=cfg["model_name"],
        hparams=cfg,
    )
