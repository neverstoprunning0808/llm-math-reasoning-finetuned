import torch
from datasets import load_dataset, DatasetDict
from src import load_config, AppConfig


def download_dataset(config: AppConfig) -> None:
    dataset = load_dataset(config.data.name, config.data.subset)

    # split train into train and val set
    split = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=8)
    new_dataset = DatasetDict({
        "train": split["train"],
        "val": split["test"],
        "test": dataset["test"]
    })

    new_dataset.save_to_disk(config.data.path)


if __name__ == "__main__":
    config = load_config()
    download_dataset(config)
