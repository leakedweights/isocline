import jax
import jax.numpy as jnp


def cast_dim(x, n_dims, start_dim=None):
    dim_diff = n_dims - x.ndim
    if start_dim is None:
        extra_dim_indices = [-i for i in range(1, dim_diff+1)]
    else:
        extra_dim_indices = [start_dim + i for i in range(dim_diff)]
    return jnp.expand_dims(x, extra_dim_indices)


def update_ema(ema_params, new_params, ema_decay):
    def _node_update(ema_node, new_node):
        return ema_decay * ema_node + (1 - ema_decay) * new_node
    updated_params = jax.tree.map(_node_update, ema_params, new_params)
    return updated_params
