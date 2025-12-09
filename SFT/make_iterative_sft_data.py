import json
import re
from pathlib import Path


INPUT_FILE = "AFM-MHQA-Agent-SFT-Dataset-Processed.json"
OUTPUT_FILE = "iterative_sft_data.json"


def split_trajectory(system_prompt: str, user_input: str, full_output: str):
    """
    Split a full output trajectory into multiple step-wise SFT samples.

    For each sample:
      - instruction: the full history so far
        (system prompt + question + previous think/search/information blocks)
      - input: always set to "" (Alpaca-style format)
      - output: the model’s next action for this step
        (e.g., <think>...</think><search>...</search> or
               <think>...</think><answer>...</answer>)
    """
    samples = []

    # The full_output is structured roughly as:
    #   <think>...</think>
    #   <search>...</search>
    #   <information>...</information>
    #   <think>...</think>
    #   <answer>...</answer>
    #
    # We split by <information> blocks to get:
    #   [model_output_1, info_1, model_output_2, info_2, ...]
    parts = re.split(r'(<information>.*?</information>)', full_output, flags=re.DOTALL)

    # Initial history contains only: system prompt + user question.
    # Note: in our data, `input` is already like "Question: xxx",
    # so we can directly concatenate it.
    current_history = f"{system_prompt}\n\n{user_input}\n"

    for i in range(0, len(parts), 2):
        model_output = parts[i].strip()
        if not model_output:
            continue

        # Create one training sample: given current history → predict model_output
        samples.append({
            "instruction": current_history,
            "input": "",
            "output": model_output
        })

        # Add the model output we just predicted into the history
        # so that the next step sees it as context.
        current_history += model_output

        # If there is an <information> block right after this output,
        # also append it to the history (this is environment feedback,
        # not something the model should generate).
        if i + 1 < len(parts):
            info_block = parts[i + 1]
            current_history += info_block

    return samples


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    all_samples = []
    for idx, item in enumerate(raw_data):
        system_prompt = item.get("instruction", "")
        user_input = item.get("input", "")
        full_output = item.get("output", "")

        step_samples = split_trajectory(system_prompt, user_input, full_output)
        all_samples.extend(step_samples)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} original examples, "
                  f"total step samples: {len(all_samples)}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(raw_data)} original examples → "
          f"{len(all_samples)} step-wise SFT samples.")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
