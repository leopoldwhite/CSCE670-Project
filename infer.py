import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL_ID = "./models/Qwen2.5-3B-Iterative-SFT-Merged"
DEFAULT_DATASET = "./data/nq_search/test.parquet"
DEFAULT_OUTPUT = "./outputs/nq_eval_results.json"
PROMPT_TEMPLATE = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
)
CURR_SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"
SEARCH_STOP = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
ANSWER_STOP = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
STOP_SEQUENCES = SEARCH_STOP + ANSWER_STOP
SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ARTICLE_PATTERN = re.compile(r"\b(a|an|the)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Search-R1 Qwen2.5 model on NQ using vLLM with Exact Match."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="HF model or checkpoint path.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to the nq_search test parquet.")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT, help="Where to store detailed predictions.")
    parser.add_argument("--retrieval-url", type=str, default="http://127.0.0.1:8000/retrieve", help="Retrieval server.")
    parser.add_argument("--retrieval-topk", type=int, default=3, help="Top-k passages to include per search call.")
    parser.add_argument("--retrieval-timeout", type=int, default=60, help="Timeout (s) for retrieval server.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value.")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling value (-1 keeps backend default).")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens per vLLM generation call.")
    parser.add_argument("--max-turns", type=int, default=8, help="Max reasoning + search turns per question.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism for vLLM.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="vLLM dtype (e.g., float16, bfloat16).")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM max model length override.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization hint.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def ensure_question_format(question: str) -> str:
    question = (question or "").strip()
    if question and question[-1] != "?":
        question += "?"
    return question


def prepare_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    user_prompt = PROMPT_TEMPLATE.format(question=ensure_question_format(question))
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return user_prompt


def get_last_query(text: str) -> Optional[str]:
    if not text:
        return None
    matches = SEARCH_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def passages_to_string(retrieval_result: List[dict]) -> str:
    formatted_reference: List[str] = []
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        parts = content.split("\n", 1)
        title = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        formatted_reference.append(f"Doc {idx + 1}(Title: {title}) {body.strip()}")
    return "\n".join(formatted_reference)


def call_search(query: str, url: str, topk: int, timeout: int) -> str:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json().get("result", [])
    if not result:
        return ""
    return passages_to_string(result[0])


def run_reasoning_with_search(
    question: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    retrieval_url: str,
    retrieval_topk: int,
    retrieval_timeout: int,
    max_turns: int,
) -> str:
    prompt = prepare_prompt(question, tokenizer)
    for turn in range(max_turns):
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        output = outputs[0].outputs[0].text
        if not output:
            break
        search_query = get_last_query(output)
        if search_query:
            search_results = call_search(search_query, retrieval_url, retrieval_topk, retrieval_timeout)
            prompt += CURR_SEARCH_TEMPLATE.format(output_text=output, search_results=search_results)
            continue
        return output
    raise RuntimeError(f"Max turns ({max_turns}) reached without producing a final answer.")


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


def answers_to_list(raw_answers) -> List[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, np.ndarray):
        raw_answers = raw_answers.tolist()
    if isinstance(raw_answers, (list, tuple)):
        return [str(ans) for ans in raw_answers]
    return [str(raw_answers)]


def compute_exact_match(prediction: str, references: List[str]) -> bool:
    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return False
    for ref in references:
        if normalize_text(ref) == pred_norm:
            return True
    return False


def main():
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().absolute()
    output_path = Path(args.output_json).expanduser().absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples)
    records = df.to_dict(orient="records")
    if not records:
        print("Dataset is empty. Nothing to evaluate.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    llm_kwargs = {
        "model": args.model_id,
        "tokenizer": args.model_id,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop=STOP_SEQUENCES,
        include_stop_str_in_output=True,
    )

    prediction_rows = []
    correct = 0
    for row in tqdm(records, desc="Evaluating NQ", total=len(records)):
        question_id = row.get("id", "")
        question = row.get("question", "")
        golden_answers = answers_to_list(row.get("golden_answers"))
        try:
            model_output = run_reasoning_with_search(
                question=question,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                retrieval_url=args.retrieval_url,
                retrieval_topk=args.retrieval_topk,
                retrieval_timeout=args.retrieval_timeout,
                max_turns=args.max_turns,
            )
        except Exception as error:
            print(f"[WARN] question {question_id} failed with error: {error}")
            model_output = ""
        predicted_answer = extract_answer(model_output)
        is_em = compute_exact_match(predicted_answer, golden_answers)
        correct += int(is_em)
        prediction_rows.append(
            {
                "question_id": question_id,
                "question": question,
                "model_output": model_output,
                "ground_truth": golden_answers,
                "prediction": predicted_answer,
                "exact_match": bool(is_em),
            }
        )

    accuracy = correct / len(prediction_rows)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(prediction_rows, f, ensure_ascii=False, indent=2)

    print(f"Exact Match Accuracy: {accuracy:.4f} ({correct}/{len(prediction_rows)})")
    print(f"Saved detailed predictions to {output_path}")


if __name__ == "__main__":
    main()