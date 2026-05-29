from zencfg import ConfigBase

from .models import ModelConfig, SimpleFNOConfig


class WandbConfig(ConfigBase):
    log: bool = True
    entity: str | None = "your_entity"
    project: str | None = "wasp-CGLE"
    name: str | None = "test-run"
    group: str | None = None


class SlurmConfig(ConfigBase):
    mem: int = 256
    timeout_hours: int = 18
    partition: str = "gpu"
    nodes: int = 1
    gres: str = "gpu"
    cpus: int = 32
    additional_parameters: dict[str, str] = {}
    setup: list[str] = []


class GLEquationConfig(ConfigBase):
    name: str = "GLEquation"
    # equation parameters
    root_dir: str = "./data/gl_datasets"
    alpha: float = 2.0
    beta: float = -2.0
    domain: tuple[float, float] = (6.4, 50.0)
    resolution: tuple[int, int] = (128, 256)
    loss_weights: dict[str, float] = {
        "sol": 10.0,
        "sol_ic": 1.0,
        "cond": 1.0,
        "cond_ic": 0.1,
        "cond_h1": 0.0,
    }
    ot_rank: int = 96
    fitting_samples: int = 5000
    test_samples: int = 1000
    ot_eps: float = 1e-3
    ot_max_iters: int = 1000
    ot_stop_thresh: float = 1e-5
    verbose: bool = True


class GLTrainerConfig(ConfigBase):
    # training config
    batch_size_buffer: int = 32
    batch_size_gen: int = 32
    batch_size_eval: int = 64
    n_steps: int = 200000
    postprocess: dict[str, int | None] = {"n_modes": None}
    max_grad_norm: float = 10000.0
    buffer_nsamples: int = 16
    eval_freq: int = 1000
    eval_num_vis: int = 4
    dtype: str = "complex64"
    save_freq: int = 10000
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    # optimizer config
    optimizer: str = "AdamW"
    lr: float = 4e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    # scheduler config
    scheduler: str = "StepLR"
    step_size: int = 20000
    gamma: float = 0.9
    # buffer config
    buffer_size: int = 20000
    buffer_gen_batch_size: int = 1000
    init_buffer_size: int = 20000
    # normalizer config
    normalizer_momentum: float = 1e-3
    # transport config
    transport_enabled: bool = True
    transport_start_s: float = 0.0
    transport_threshold: float = 0.3
    transport_increment: float = 0.002
    transport_min_sample: int = 1000
    transport_base_schedule: str = "cosine"
    transport_adaptive_range: tuple[float, float] = (0.0, 0.9)
    transport_pre_training_ratio: float = 0.01


class FNO_2D(SimpleFNOConfig):
    data_channels: int = 5
    out_channels: int = 2
    n_modes: list[int] = [64, 64]
    hidden_channels: int = 64
    projection_channel_ratio: int = 2
    domain_padding: list[float] = [0.1, 0]
    norm: str = "group_norm"
    complex_data: bool = False


class DistributedConfig(ConfigBase):
    use_distributed: bool = False
    model_parallel_size: int | None = 1
    seed: int | None = 1


class GLDefault(ConfigBase):
    verbose: bool = True
    debug: bool = False
    arch: str = "fno"
    submit: bool = False
    slurm: SlurmConfig = SlurmConfig()
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_2D()
    equation: GLEquationConfig = GLEquationConfig()
    trainer: GLTrainerConfig = GLTrainerConfig()
    wandb: WandbConfig = WandbConfig()
