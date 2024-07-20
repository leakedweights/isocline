import flax.linen as nn

consistency_config = {
    "sigma_data": 0.5,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "huber_const": 0.00054,
    "rho": 7.0,
    "p_mean": 1.1,
    "p_std": 2.0,
    "s0": 10,
    "s1": 1280
}


model_config = {
    "nonlinearity": nn.swish,
    "channel_mults": (1, 2, 4, 8, 16),
    "attention_mults": (2, 8),
    "kernel_size": (3, 3),
    "num_init_channels": 16,
    "num_res_blocks": 3,
    "pos_emb_type": "sinusoidal",
    "pos_emb_dim": 512,
    "rescale_skip_conns": True,
    "resblock_variant": "BigGAN++",
    "dropout": 0.2,
    "use_context": True,
    "context_fn": lambda x: x
}

trainer_config = {
    "learning_rate": 2e-4,
    "use_ema": False,

    "optimizer": None,
    "log_wandb": True,
    "log_frequency": 100,

    "create_snapshots": True,
    "snapshot_frequency": 10_000,
    "samples_to_keep": 5,
    "snapshot_dir": None,

    "run_evals": True,
    "eval_frequency": 5000,
    "ground_truth_dir": None,
    "eval_dir": None,
    "num_eval_samples": int(3e4),

    "fid_params": {
        "device": None,
        "verbose": False
    },

    "checkpoint_frequency": 50_000,
    "checkpoints_to_keep": 10,
    "checkpoint_dir": None
}
