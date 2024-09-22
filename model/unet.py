import jax
from jax import numpy as jnp
from flax import linen as nn


# depthwise separable convolution
class DepthwiseSeparableConv(nn.Module):
    features: int = 64
    kernel_size: int = 5
    strides: int = 1
    padding: str = 'SAME'
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        # x = nn.relu(x)
        return x


class PositionalEmbedding(nn.Module):
    emb_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.emb_dim // 2,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.kaiming_normal()
        )(x) * jnp.pi
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        return x


# coordConv
class CoordConv(nn.Module):
    def __call__(self, x):
        batch_size, height, width, channels = x.shape
        yv = jnp.arange(0, height).reshape((1, height, 1, 1))
        xv = jnp.arange(0, width).reshape((1, 1, width, 1))
        yv = jnp.tile(yv, (batch_size, 1, width, 1))
        xv = jnp.tile(xv, (batch_size, height, 1, 1))
        coord_feature = jnp.concatenate([x, xv, yv], axis=-1)
        return coord_feature


class ConvBlock(nn.Module):
    features: int = 64
    n: int = 2
    training: bool = True

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n):
            x = nn.Conv(self.features,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                use_bias=False,
                kernel_init=nn.initializers.kaiming_normal()
            )(x)
            x = nn.BatchNorm(use_running_average=not self.training)(x)
            x = nn.relu(x)
        return x


class ConvUpsample(nn.Module):
    features: int = 64
    upsample: int = 2

    @nn.compact
    def __call__(self, x):
        # x = PositionalEmbedding(self.features)(x)
        x = CoordConv()(x)
        if self.upsample > 1:
            x = jax.image.resize(x,
                shape=(x.shape[0], x.shape[1] * self.upsample, x.shape[2] * self.upsample, x.shape[3]),
                method='bilinear'
            )
        x = nn.Conv(self.features,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        x = nn.sigmoid(x)
        return x


class Encoder(nn.Module):
    features: int = 64
    training: bool = True

    @nn.compact
    def __call__(self, x):
        e1 = ConvBlock(self.features, training=self.training)(x)
        e1_pool = nn.max_pool(e1, window_shape=(2, 2), strides=(2, 2))

        e2 = ConvBlock(self.features * 2, training=self.training)(e1_pool)
        e2_pool = nn.max_pool(e2, window_shape=(2, 2), strides=(2, 2))

        e3 = ConvBlock(self.features * 4, training=self.training)(e2_pool)
        e3_pool = nn.max_pool(e3, window_shape=(2, 2), strides=(2, 2))

        e4 = ConvBlock(self.features * 8, training=self.training)(e3_pool)
        e4_pool = nn.max_pool(e4, window_shape=(2, 2), strides=(2, 2))

        e5 = ConvBlock(self.features * 16, training=self.training)(e4_pool)

        return e1, e2, e3, e4, e5


class Decoder(nn.Module):
    features: int = 64
    max_mask: int = 32
    training: bool = True

    @nn.compact
    def __call__(self, e1, e2, e3, e4, e5):
        up_chans = self.features * 4

        e3_d4 = nn.max_pool(e3, window_shape=(2, 2), strides=(2, 2))
        e3_d4 = ConvBlock(self.features, n=1, training=self.training)(e3_d4)
        e4_d4 = ConvBlock(self.features, n=1, training=self.training)(e4)
        e5_d4 = jax.image.resize(e5, shape=(e5.shape[0], e5.shape[1] * 2, e5.shape[2] * 2, e5.shape[3]), method='bilinear')
        e5_d4 = ConvBlock(self.features, n=1, training=self.training)(e5_d4)
        d4 = jnp.concatenate([e3_d4, e4_d4, e5_d4], axis=-1)
        d4 = ConvBlock(up_chans, n=1, training=self.training)(d4)

        e2_d3 = nn.max_pool(e2, window_shape=(2, 2), strides=(2, 2))
        e2_d3 = ConvBlock(self.features, n=1, training=self.training)(e2_d3)
        e3_d3 = ConvBlock(self.features, n=1, training=self.training)(e3)
        e4_d3 = jax.image.resize(e4, shape=(e4.shape[0], e4.shape[1] * 2, e4.shape[2] * 2, e4.shape[3]), method='bilinear')
        e4_d3 = ConvBlock(self.features, n=1, training=self.training)(e4_d3)
        d3 = jnp.concatenate([e2_d3, e3_d3, e4_d3], axis=-1)
        d3 = ConvBlock(up_chans, n=1, training=self.training)(d3)

        e1_d2 = nn.max_pool(e1, window_shape=(2, 2), strides=(2, 2))
        e1_d2 = ConvBlock(self.features, n=1, training=self.training)(e1_d2)
        e2_d2 = ConvBlock(self.features, n=1, training=self.training)(e2)
        d3_d2 = jax.image.resize(d3, shape=(d3.shape[0], d3.shape[1] * 2, d3.shape[2] * 2, d3.shape[3]), method='bilinear')
        d3_d2 = ConvBlock(self.features, n=1, training=self.training)(d3_d2)
        d2 = jnp.concatenate([e1_d2, e2_d2, d3_d2], axis=-1)
        d2 = ConvBlock(up_chans, n=1, training=self.training)(d2)

        e1_d1 = ConvBlock(self.features, n=1, training=self.training)(e1)
        d2_d1 = jax.image.resize(d2, shape=(d2.shape[0], d2.shape[1] * 2, d2.shape[2] * 2, d2.shape[3]), method='bilinear')
        d2_d1 = ConvBlock(self.features, n=1, training=self.training)(d2_d1)
        d3_d1 = jax.image.resize(d3, shape=(d3.shape[0], d3.shape[1] * 4, d3.shape[2] * 4, d3.shape[3]), method='bilinear')
        d3_d1 = ConvBlock(self.features, n=1, training=self.training)(d3_d1)
        d4_d1 = jax.image.resize(d4, shape=(d4.shape[0], d4.shape[1] * 8, d4.shape[2] * 8, d4.shape[3]), method='bilinear')
        d4_d1 = ConvBlock(self.features, n=1, training=self.training)(d4_d1)
        d1 = jnp.concatenate([e1_d1, d2_d1, d3_d1, d4_d1], axis=-1)
        d1 = ConvBlock(up_chans, n=1, training=self.training)(d1)

        # branch for keypoints
        hmap = ConvBlock(self.features * 2, n=1, training=self.training)(d1)
        d5 = ConvUpsample(self.features * 2, upsample=16)(e5)
        branch_hmap = hmap + d5
        hmap = nn.Conv(1,
            kernel_size=(1, 1),
            strides=1,
            kernel_init=nn.initializers.kaiming_normal(),
        )(hmap)

        # branch for charmap
        mask = ConvBlock(self.features * 2, n=1, training=self.training)(d1)
        mask = branch_hmap * mask
        mask = nn.Conv(self.max_mask,
            kernel_size=(1, 1),
            strides=1,
            kernel_init=nn.initializers.kaiming_normal(),
        )(mask)

        return mask, hmap

        # # supervision
        # d1 = ConvUpsample(self.ord_nums, upsample=0)(d1)

        # # if self.training:
        # d2 = ConvUpsample(self.max_mask, upsample=2)(d2)
        # d3 = ConvUpsample(self.max_mask, upsample=4)(d3)
        # d4 = ConvUpsample(self.max_mask, upsample=8)(d4)
        # d5 = ConvUpsample(self.max_mask, upsample=16)(e5)
        # return mask, (d1, d2, d3, d4, d5)

        # return char, d1


class UNetV3(nn.Module):
    features: int = 32
    max_mask: int = 32

    @nn.compact
    def __call__(self, x, train=True):
        z1, z2, z3, z4, z5 = Encoder(
            self.features,
            training=train,
        )(x)
        y = Decoder(
            self.features,
            max_mask=self.max_mask,
            training=train,
        )(z1, z2, z3, z4, z5)
        return y


if __name__ == '__main__':
    # jax cpu
    jax.config.update("jax_platform_name", "cpu")

    key = jax.random.PRNGKey(0)
    model = UNetV3(features=16, max_mask=32)
    x = jnp.ones((1, 256, 256, 1))
    params = model.init(key, x)
    out, batch_stats = model.apply(
        params, x,
        mutable=['batch_stats'],
        rngs={'dropout': key}
    )

    table_fn = nn.tabulate(
        model,
        key,
        compute_flops=True,
        compute_vjp_flops=True,
    )
    print(table_fn(x))

    for y in out:
        print(y.shape)
