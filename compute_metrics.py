#!/usr/bin/env python3
"""Utility to compute Exact Match and ROUGE-L from saved prediction JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

DEFAULT_PREDICTIONS = "./outputs/nq_eval_searchr1_results.json"
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b")
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Exact Match and ROUGE-L from infer.py outputs."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=DEFAULT_PREDICTIONS,
        help="Path to the JSON file produced by infer.py/run_nq_eval.sh.",
    )
    return parser.parse_args()


def answers_to_list(raw_answers) -> List[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        return [str(ans) for ans in raw_answers]
    return [str(raw_answers)]


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = ARTICLE_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_answer(text: str) -> str:
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def compute_exact_match(prediction: str, references: Iterable[str]) -> bool:
    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return False
    for ref in references:
        if normalize_text(ref) == pred_norm:
            return True
    return False


def tokenize_for_rouge(text: str) -> List[str]:
    return (text or "").strip().lower().split()


def lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    if not seq_a or not seq_b:
        return 0
    dp = [0] * (len(seq_b) + 1)
    for token_a in seq_a:
        prev = 0
        for j, token_b in enumerate(seq_b):
            temp = dp[j + 1]
            if token_a == token_b:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = temp
    return dp[-1]


def rouge_l_score(prediction: str, references: Iterable[str]) -> float:
    pred_tokens = tokenize_for_rouge(prediction)
    if not pred_tokens:
        return 0.0
    best_score = 0.0
    for reference in references:
        ref_tokens = tokenize_for_rouge(reference)
        if not ref_tokens:
            continue
        lcs = lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            continue
        score = (2 * precision * recall) / (precision + recall)
        best_score = max(best_score, score)
    return best_score


def pick_prediction(entry: dict) -> str:
    prediction = entry.get("prediction")
    if prediction is not None:
        return str(prediction)
    return extract_answer(entry.get("model_output", ""))


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions).expanduser().absolute()
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} does not exist.")

    with predictions_path.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    if not isinstance(records, list):
        raise ValueError("Prediction JSON must contain a list of records.")

    if not records:
        print("No predictions found; nothing to score.")
        return

    total = len(records)
    em_hits = 0
    rouge_sum = 0.0

    for entry in records:
        prediction = pick_prediction(entry)
        references = answers_to_list(entry.get("ground_truth") or entry.get("golden_answers"))
        em_hits += int(compute_exact_match(prediction, references))
        rouge_sum += rouge_l_score(prediction, references)

    em_score = em_hits / total
    rouge_l = rouge_sum / total

    print(f"Loaded {total} predictions from {predictions_path}")
    print(f"Exact Match: {em_score:.4f} ({em_hits}/{total})")
    print(f"Average ROUGE-L: {rouge_l:.4f}")


if __name__ == "__main__":
    main()


