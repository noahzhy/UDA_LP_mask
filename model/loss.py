import optax
import jax
import jax.numpy as jnp


# mIoU acc
@jax.jit
def mIoU_acc(pred, target):
    pred = jax.nn.sigmoid(pred)
    pred = jnp.round(pred)
    inter = jnp.sum(pred * target)
    union = jnp.sum(pred) + jnp.sum(target)
    return inter / (union - inter)


# tversky loss
@jax.jit
def tversky_loss(logits, targets, beta=0.7, smooth=1e-7):
    alpha = 1 - beta
    pred = jax.nn.sigmoid(logits).flatten()
    true = targets.flatten()
    tp = jnp.sum(pred * true)
    fp = jnp.sum(pred * (1 - true))
    fn = jnp.sum((1 - pred) * true)
    loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return loss.mean()


# Binary focal loss for heatmap calculation
@jax.jit
def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    pred = jax.nn.sigmoid(pred)
    _zeros = jnp.zeros_like(pred)
    pos_p_sub = jnp.where(target > _zeros, target - pred, _zeros)
    neg_p_sub = jnp.where(target > _zeros, _zeros, pred)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * jnp.log(jnp.clip(pred, 1e-8, 1.0)) - (1 - alpha) * (neg_p_sub ** gamma) * jnp.log(jnp.clip(1.0 - pred, 1e-8, 1.0))
    return jnp.mean(per_entry_cross_ent)


# test mIoU acc on random data with one hot encoding
def test_mIoU_acc():
    pred = jnp.array([[[[0., 1.], [1., 0.]]]])
    target = jnp.array([[[[1., 0.], [1, 0.]]]])
    print(mIoU_acc(pred, target))


if __name__ == "__main__":
    test_mIoU_acc()
    print("All tests passed")
    y_true = jnp.array([1., 0., 1., 1., 0.])
    y_pred = jnp.array([0.99, 0.1, 0.8, 0.9, 0.1])
    loss_value = focal_loss(y_true, y_pred)
    print(loss_value)