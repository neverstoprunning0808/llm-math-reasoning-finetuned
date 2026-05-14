from pydantic import BaseModel, Field
import yaml


class DataConfig(BaseModel):
    path: str
    name: str
    subset: str

class ModelConfig(BaseModel):
    model_checkpoint: str
    tokenizer_checkpoint: str
    num_return_sequences: int
    temperature: float
    batch_size: int

class CoTConfig(BaseModel):
    num_sample: int

class DirectSFTConfig(BaseModel):
    mode: str
    lora_rank: int
    lora_alpha: int
    bias: str
    target_modules: str
    task_type: str
    output_dir: str
    report_to: str
    gradient_checkpointing: float
    learning_rate: float
    per_device_train_batch_size: int
    num_train_epochs: int
    logging_steps: int
    weight_decay: float
    save_strategy: str
    evaluate_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool


class COTSFTConfig(BaseModel):
    mode: str
    lora_rank: int
    lora_alpha: int
    bias: str
    target_modules: str
    task_type: str
    output_dir: str
    report_to: str
    gradient_checkpointing: float
    learning_rate: float
    per_device_train_batch_size: int
    num_train_epochs: int
    logging_steps: int
    weight_decay: float
    save_strategy: str
    evaluate_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool


class AppConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    cot: CoTConfig
    directsft: DirectSFTConfig
    cotsft: COTSFTConfig
    # dataloader: DataLoader
    # train: TrainConfig


def load_config(path:str = "params.yaml") -> AppConfig:
    with open(path) as f:
        config = yaml.safe_load(f)

    return AppConfig(**config)


if __name__ == "__main__":
    load_config()