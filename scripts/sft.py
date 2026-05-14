from src import LLMModel, split_data, extract_answer, accuracy, tokenizer_function
from peft  import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer, TrainingArguments
from src import load_config, AppConfig
from datasets import load_from_disk
from functools import partial


def train_direct_sft_model(config: AppConfig):
    
    # load the model:
    base_model = LLMModel(config)

    # lora config:
    lora_config = LoraConfig(
        r = config.directsft.lora_rank,
        lora_alpha = config.directsft.lora_alpha,
        target_modules = config.directsft.target_modules,
        bias = config.directsft.bias,
        task_type = config.directsft.task_type
    )

    # apply lora config to the model:
    base_model.model = get_peft_model(base_model.model, lora_config).to(base_model.device)
    
    # base_model.model.enable_input_require_grads()


    # training args
    training_args = TrainingArguments(
        output_dir = config.directsft.output_dir,
        report_to = config.directsft.report_to,
        learning_rate = config.directsft.learning_rate,
        gradient_checkpointing = config.directsft.gradient_checkpointing,
        per_device_train_batch_size = config.directsft.per_device_train_batch_size,
        num_train_epochs = config.directsft.num_train_epochs,
        weight_decay = config.directsft.weight_decay,
        save_strategy = config.directsft.save_strategy,
        # eval_strategy = config.directsft.evaluate_strategy,
        logging_steps = config.directsft.logging_steps,
        # load_best_model_at_end = config.directsft.load_best_model_at_end,
        save_total_limit=2,
        # metric_for_best_model=config.directsft.metric_for_best_model,
        # greater_is_better=config.directsft.greater_is_better,
    )

    # create tokenized data
    data = load_from_disk(config.data.path)

    train_data = data['train'].map(partial(tokenizer_function,
                                    tokenizer = base_model.tokenizer,
                                    mode = config.directsft.mode),
                                    batched=True,
                                    batch_size=config.model.batch_size,
                                    load_from_cache_file=False,
                                    remove_columns=['question', 'answer']
                                )


    valid_data = data['val'].map(partial(tokenizer_function,
                                    tokenizer = base_model.tokenizer,
                                    mode = config.directsft.mode),
                                    batched=True,
                                    batch_size=config.model.batch_size,
                                    load_from_cache_file=False,
                                    remove_columns=['question', 'answer']
                                )
    
    # create trainer
    trainer = Trainer(
        model = base_model.model,
        args = training_args,
        train_dataset = train_data,
        eval_dataset = valid_data
    )

    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(config.directsft.output_dir)

    # tensorboard --logdir directsft/runs    



if __name__ == "__main__":
    config = load_config()
    train_direct_sft_model(config)



