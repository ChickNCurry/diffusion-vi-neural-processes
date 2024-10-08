from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class Stringifiable:
    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig(Stringifiable):
    x_dim: int
    y_dim: int
    r_dim: Optional[int]
    z_dim: Optional[int]
    h_dim: int
    num_layers_det_enc: Optional[int]
    num_layers_lat_enc: Optional[int]
    num_layers_dec: int
    non_linearity: str
    is_attentive: bool
    aggregation: str
    fixed_start_density: Optional[bool]
    diffusion_process: Optional[str]
    num_steps: Optional[int]
    num_layers_diffu_process: Optional[int]


@dataclass
class DataConfig(Stringifiable):
    benchmark: str
    n_task: int
    n_datapoints_per_task: int
    output_noise: float
    seed_task: int
    seed_x: int
    seed_noise: int


@dataclass
class TrainValConfig(Stringifiable):
    num_epochs: int
    batch_size: int
    learning_rate: float
    split: Tuple[float, float]


@dataclass
class Config(Stringifiable):
    data_config: DataConfig
    model_config: ModelConfig
    train_val_config: TrainValConfig
