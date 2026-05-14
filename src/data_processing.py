from datasets import DatasetDict
from src import LLMModel, AppConfig, load_config, extract_answer
from datasets import load_from_disk
from transformers import PreTrainedTokenizerBase
from typing import Literal


def split_data(data: DatasetDict) -> tuple[list[str], list[str], list[str], list[str]]:
    train = data['train']
    X_train, y_train = train['question'], train['answer']
    
    valid = data['val']
    X_valid, y_valid = valid['question'], valid['answer']

    test = data['test']
    X_test, y_test = test['question'], test['answer']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def _tokenize(tokenizer: PreTrainedTokenizerBase, mode: Literal["cot_sft", "direct_answer_sft"], question: str, answer: str):

    if mode not in ["cot_sft", "direct_answer_sft"]:
        raise ValueError("must be 'cot_sft' OR 'direct_answer_sft']")
    
    # format the answer for direct_answer_sft ~ get only the int number
    if mode == "direct_answer_sft":
        answer = extract_answer([answer])[0]
        answer = f"#### {answer}"
    
    tokenizer.padding_side = "right" # this is to mask the answer from the question later and by default
    tokenizer.pad_token = tokenizer.eos_token

    merge_qa = f"{question} {answer} {tokenizer.eos_token}" # combine q & a

    input_with_padding = tokenizer(merge_qa, padding="max_length", truncation=True, max_length=300)

    input_with_padding_ids = input_with_padding["input_ids"]


    # label mask for loss calculation: Pytorch ignore index = -100:
    # mask the question: use the question len
    question_len = len(tokenizer(question)["input_ids"])
    labels = [-100] * question_len + input_with_padding_ids[question_len:]

    # mask the padding: use the attention_mask = 0
    for i in range(len(labels)):
        if input_with_padding['attention_mask'][i] == 0: # if it's the padding token -> mask
            labels[i] = -100


    input_with_padding['labels'] = labels

    return input_with_padding


def tokenizer_function(examples: dict[str, list], tokenizer: PreTrainedTokenizerBase, mode: str):
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for question, answer in zip(examples["question"], examples["answer"]):
        tokenized = _tokenize(tokenizer, mode, question, answer)
        results["input_ids"].append(tokenized["input_ids"])
        results["attention_mask"].append(tokenized["attention_mask"])
        results["labels"].append(tokenized["labels"])
    
    return results


# from functools import partial

# if __name__ == "__main__":
#     config = load_config()
#     model = LLMModel(config)


#     data = load_from_disk(config.data.path)
    # _, _, X_valid, y_valid = split_data(data)
    # val_data = data['test']

    # input_with_padding = tokenize(model.tokenizer, X_valid[0], y_valid[0])


    # print("---------------")
    # print(input_with_padding['input_ids'])
    # print("---------------")
    # print(input_with_padding['attention_mask'])
    # print("---------------")
    # print(input_with_padding['labels'])
    
    # mode = "direct_answer_sft"
    # val_data = data['test']

    # tokenized_val_data = val_data.map(
    #     partial(tokenizer_function, mode=mode, tokenizer=model.tokenizer),
    #     batched=True,
    #     batch_size=config.model.batch_size,
    #     load_from_cache_file=False,
    #     remove_columns=['question', 'answer']
    # )

    # print(tokenized_val_data)

    # print(model.generate(tokenized_val_data['question'][0]))