from src import AppConfig, LLMModel, load_config, accuracy, extract_answer, format_accuracy, extract_generated_answer
from datasets import load_from_disk
from peft import PeftModel
import os

def test_direct_sft_model(config: AppConfig):
    data = load_from_disk(config.data.path)
    test_data = data['test']

    base_model = LLMModel(config)
    base_model.model = PeftModel.from_pretrained(base_model.model, config.cotsft.output_dir).to(base_model.model.device)
    
    print(base_model.device)

    base_model.model.eval()

    test_question = test_data['question']
    test_answer = test_data['answer']

    generated_answers = base_model.batch_generate(test_question)
    preds_answer = extract_answer(generated_answers)
    truth_answer = extract_answer(test_answer)

    for q, a in zip(test_question[:3], truth_answer[:3]):
        gen_a = base_model.batch_generate([q])
        print(a)
        print(extract_answer(gen_a))


    pred_acc = accuracy(preds_answer, truth_answer)
    format_acc = format_accuracy(preds_answer)

    print(f"Accuracy: {pred_acc:.4f}")
    print(f"Format accuracy: {format_acc:.4f}")


    metrics = {
        "format_acc": format_acc,
        "accuracy_score": pred_acc
    }

    metrics_path = "metrics/cotsft_metrics_eval.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        metrics.dump(metrics, f, indent=4)


if __name__ == "__main__":
    config = load_config()

    test_direct_sft_model(config)
