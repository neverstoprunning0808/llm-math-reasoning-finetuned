import torch
import math

def extract_answer(reasonings: list[str]) -> int:
    results = []
    for text in reasonings:
        try:
            value = int(text.split("####")[1].strip())
        except(IndexError, ValueError):
            value = float('inf')
        results.append(value)
    return results


def extract_generated_answer(preds: list[str]) -> int:
    results = []
    for text in preds:
        try:
            value = int(text.strip())
        except(IndexError, ValueError):
            value = float('inf')
        results.append(value)
    return results


def accuracy(truth: list[int|float], preds: list[int|float]) -> float:
    n = len(truth)
    is_correct = sum((torch.tensor(truth) == torch.tensor(preds)).int()).item()

    return is_correct / n


def format_accuracy(preds: list[int|float]) -> float:
    n = len(preds)
    is_wrong_format = sum(math.isinf(x) for x in preds)

    return 1 - is_wrong_format/n