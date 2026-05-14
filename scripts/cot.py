import torch
from src import load_config, AppConfig
from src import LLMModel, accuracy, extract_answer, split_data, format_accuracy
from datasets import load_from_disk
from pprint import pprint
import os
import json

def cot_experiment(config: AppConfig) -> None:
    config.model.num_return_sequences = 1 # just do 1 pred

    model = LLMModel(config)

    data = load_from_disk(config.data.path) 

    _, _, _, _, X_test, y_test = split_data(data)
    X_test, y_test = X_test[:config.cot.num_sample], y_test[:config.cot.num_sample]

    truth_answer = extract_answer(y_test)

    prompts = [model.format_cot_template(question) for question in X_test]

    generated_answer = model.batch_generate(prompts)
    pprint(generated_answer[:10])

    pred_answers = extract_answer(generated_answer)
    print(pred_answers)

    format_pred_acc = format_accuracy(pred_answers)
    print(f"Format accuracy: {format_pred_acc:.4f}")

    accuracy_score = accuracy(truth_answer, pred_answers)
    print(f"Accuracy score: {accuracy_score:.4f}")
    
    metrics = {
        "format_acc": format_pred_acc,
        "accuracy_score": accuracy_score
    }

    metrics_path = "metrics/cot_metrics_eval.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    


if __name__ == "__main__":
    config = load_config()

    cot_experiment(config)
    

