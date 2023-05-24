import equinox as eqx
import jax
from jax.random import split


class ResnetMLP(eqx.Module):
    in_linear: eqx.nn.Linear
    out_linear: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        *,
        width_size,
        in_size,
        out_size,
        dropout_rate,
        key,
    ):
        ks = split(key, 2)
        self.in_linear = eqx.nn.Linear(in_features=in_size, out_features=width_size, key=ks[0])
        self.out_linear = eqx.nn.Linear(in_features=width_size, out_features=out_size, key=ks[1])
        self.layer_norm = eqx.nn.LayerNorm(out_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x, *, key):
        x_ = self.in_linear(x)
        x_ = jax.nn.gelu(x_)
        x_ = self.out_linear(x_)
        x_ = self.dropout(x_, key=key)

        x = x + x_
        x = self.layer_norm(x)
        return x