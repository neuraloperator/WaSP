import math

from zencfg import ConfigBase

from .models import ModelConfig, SimpleFNOConfig


class WandbConfig(ConfigBase):
    log: bool = True
    entity: str | None = "your_entity"
    project: str | None = "wasp-ac"
    name: str | None = "test-run"
    group: str | None = None


class SlurmConfig(ConfigBase):
    mem: int = 256
    timeout_hours: int = 3
    partition: str = "gpu"
    nodes: int = 1
    gres: str = "gpu"
    cpus: int = 32
    additional_parameters: dict[str, str] = {}
    setup: list[str] = []


class ACEquationConfig(ConfigBase):
    name: str = "ACEquation"
    # equation parameters
    root_dir: str = "./data/ac_datasets"
    nu: float = 0.01
    ic_amplitude: float = 1e-3
    power_decay: float = 1
    domain: tuple[float, float] = (5, 2 * math.pi)
    resolution: tuple[int, int] = (128, 128)
    loss_weights: dict[str, float] = {
        "sol": 10.0,
        "sol_ic": 1.0,
        "cond": 1.0,
        "cond_ic": 0.1,
        "cond_h1": 0.0,
    }
    test_samples: int = 1000


class ACTrainerConfig(ConfigBase):
    # training config
    batch_size_buffer: int = 32
    batch_size_gen: int = 32
    batch_size_eval: int = 64
    n_steps: int = 20000
    postprocess: dict[str, int | None] = {}
    max_grad_norm: float = 5000.0
    buffer_nsamples: int = 32
    eval_freq: int = 1000
    eval_num_vis: int = 4
    dtype: str = "float32"
    save_freq: int = 2000
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    # optimizer config
    optimizer: str = "AdamW"
    lr: float = 4e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    # scheduler config
    scheduler: str = "StepLR"
    step_size: int = 2000
    gamma: float = 0.9
    # buffer config
    buffer_size: int = 5000
    buffer_gen_batch_size: int = 1000
    init_buffer_size: int = 5000
    # normalizer config
    normalizer_momentum: float = 1e-3
    # transport config
    transport_enabled: bool = True
    transport_start_s: float = 0.0
    transport_threshold: float = 0.2
    transport_increment: float = 0.005
    transport_min_sample: int = 1000
    transport_base_schedule: str = "cosine"
    transport_adaptive_range: tuple[float, float] = (0.0, 0.9)
    transport_pre_training_ratio: float = 0.01


class FNO_2D(SimpleFNOConfig):
    data_channels: int = 3
    out_channels: int = 1
    n_modes: list[int] = [64, 128]
    hidden_channels: int = 32
    projection_channel_ratio: int = 2
    domain_padding: list[float] = [0.1, 0]
    norm: str = "group_norm"


class DistributedConfig(ConfigBase):
    use_distributed: bool = False
    model_parallel_size: int | None = 1
    seed: int | None = 1


class ACDefault(ConfigBase):
    verbose: bool = True
    debug: bool = False
    arch: str = "fno"
    submit: bool = False
    slurm: SlurmConfig = SlurmConfig()
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_2D()
    equation: ACEquationConfig = ACEquationConfig()
    trainer: ACTrainerConfig = ACTrainerConfig()
    wandb: WandbConfig = WandbConfig()
