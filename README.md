# GSM8K Math Reasoning — LLM Fine-tuning

Experiments on fine-tuning a small LLM for grade-school math reasoning using the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset. Three approaches are implemented and compared: Chain-of-Thought (CoT) prompting, Direct Answer SFT, and CoT SFT.

---

## Project Structure

```
├── src/
│   ├── config.py               # Pydantic config with params.yaml
│   ├── model.py                # LLMModel: load, generate, CoT template
│   ├── data_processing.py      # Tokenization and data splitting
│   ├── utils.py                # accuracy, extract_answer, format_accuracy
│   └── __init__.py
├── scripts/
│   ├── prepare_data.py         # Download and split GSM8K
│   ├── cot.py                  # CoT + Instruction + Few-shot prompting experiment (no fine-tuning)
│   ├── sft.py                  # LoRA fine-tuning (direct answer)
│   ├── evaluate_direct_sft.py  # Evaluate fine-tuned direct sft model
│   ├── cot_sft.py              # LoRA fine-tuning (CoT answer)
│   └── evaluate_cot_sft.py     # Evaluate fine-tuned CoT sft model
├── requirements.txt            # required packages for installation
└── params.yaml                 # All hyperparameters
```

---

## Dataset

**GSM8K** (`openai/gsm8k`, subset: `main`) — 8.5K grade-school math word problems with step-by-step solutions and final answers marked by `####`.

The dataset is split into:
- `train` — 90% of original train set
- `val` — 10% of original train set
- `test` — original GSM8K test set

---


## Setup

```bash
pip install -r requirements.txt
git init 
dvc init 
```

Download and prepare the dataset:

```bash
python -m scripts.prepare_data
```

---

## Approaches

### 1. Chain-of-Thought (CoT) Prompting
The model is given a system prompt and few-shot examples that demonstrate step-by-step reasoning before the final answer.

```bash
python -m scripts.cot
```

### 2. Direct Answer SFT
Fine-tune the model with LoRA to directly output the final integer answer, without reasoning steps.

```bash
python -m scripts.sft
python -m scripts.evaluate_direct_sft
```

### 3. CoT SFT
Fine-tune the model with LoRA on full chain-of-thought reasoning (steps + `#### answer`).

```bash
python -m scripts.cot_sft 
python -m scripts.evaluate_cot_sft
```


---

## Configuration

All hyperparameters are managed in `params.yaml`:

---

## Metrics Evaluation

### Metrics reported:
- **Pred Accuracy** — % of predictions matching the ground truth answer
- **Format Accuracy** — % of outputs that contain a parseable answer (not `inf`)

### Chain-of-Thoughts + Few-shot + Instruction prompting:
- **Pred Accuracy** — 0.3450
- **Format Accuracy** — 0.4560

### Direct Answer SFT:
- **Pred Accuracy** — 0.1448
- **Format Accuracy** — 0.9886

### Chain-of-Thoughts SFT:
- **Pred Accuracy** — 0.5011
- **Format Accuracy** — 0.946

---

## Implementation Notes

- **LoRA fine-tuning** via `peft` — only a small number of adapter weights are trained, keeping memory usage low
- **Label masking** — the question tokens are masked with `-100` so loss is only computed on the answer portion
- **`apply_chat_template`** — uses the model's built-in chat format for consistent instruction following
- **Answer extraction** — parses `#### <number>` from model output; falls back to the last integer in the output if `####` is not present
