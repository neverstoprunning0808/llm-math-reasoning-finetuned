from src import LLMModel, split_data, extract_answer, accuracy, tokenizer_function
from peft  import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer, TrainingArguments
from src import load_config, AppConfig
from datasets import load_from_disk
from functools import partial


def train_cot_sft_model(config: AppConfig):
    
    # load the model:
    base_model = LLMModel(config)

    # lora config:
    lora_config = LoraConfig(
        r = config.cotsft.lora_rank,
        lora_alpha = config.cotsft.lora_alpha,
        target_modules = config.cotsft.target_modules,
        bias = config.cotsft.bias,
        task_type = config.cotsft.task_type
    )

    # apply lora config to the model:
    base_model.model = get_peft_model(base_model.model, lora_config).to(base_model.device)
    
    # base_model.model.enable_input_require_grads()


    # training args
    training_args = TrainingArguments(
        output_dir = config.cotsft.output_dir,
        report_to = config.cotsft.report_to,
        learning_rate = config.cotsft.learning_rate,
        gradient_checkpointing = config.cotsft.gradient_checkpointing,
        per_device_train_batch_size = config.cotsft.per_device_train_batch_size,
        num_train_epochs = config.cotsft.num_train_epochs,
        weight_decay = config.cotsft.weight_decay,
        save_strategy = config.cotsft.save_strategy,
        # eval_strategy = config.cotsft.evaluate_strategy,
        logging_steps = config.cotsft.logging_steps,
        # load_best_model_at_end = config.cotsft.load_best_model_at_end,
        save_total_limit=2,
        metric_for_best_model=config.cotsft.metric_for_best_model,
        greater_is_better=config.cotsft.greater_is_better,
    )

    # create tokenized data
    data = load_from_disk(config.data.path)

    train_data = data['train'].map(partial(tokenizer_function,
                                    tokenizer = base_model.tokenizer,
                                    mode = config.cotsft.mode),
                                    batched=True,
                                    batch_size=config.model.batch_size,
                                    load_from_cache_file=False,
                                    remove_columns=['question', 'answer']
                                )


    valid_data = data['val'].map(partial(tokenizer_function,
                                    tokenizer = base_model.tokenizer,
                                    mode = config.cotsft.mode),
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

    trainer.save_model(config.cotsft.output_dir)

    # tensorboard --logdir cotsft/runs    



if __name__ == "__main__":
    config = load_config()
    train_cot_sft_model(config)



