from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4

@dataclass
class DataConfig:
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    train_test_split: float = 0.2
    max_seq_length: int = 512

@dataclass
class TrainingConfig:
    output_dir: Path = Path("models/checkpoints")
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()