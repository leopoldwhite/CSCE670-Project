**Overview**

This folder contains the Supervised Fine-Tuning (SFT) checkpoint for our project’s student model.
The model is intended to serve as the initial policy for the RL stage (PPO/GRPO).

Base model used: Qwen/Qwen2.5-3B-Instruct

Fine-tuning method: LoRA + LLaMA-Factory

Output: LoRA adapter checkpoint



**Data Preparation**

The original teacher data was provided in: AFM-MHQA-Agent-SFT-Dataset-Processed.json

This file contained full multi-step trajectories in the format:

• &lt;think&gt;...&lt;think&gt;

• &lt;search&gt;...&lt;search&gt;

• &lt;information&gt;...&lt;information&gt;

• &lt;answer&gt;...&lt;answer&gt;

To make the dataset usable for step-wise SFT, I wrote a preprocessing script (make_iterative_sft_data.py) that:

1. Split each full trajectory into multiple training samples

2. Each sample represents only the next model action, following the iterative loop:

    • &lt;think&gt; + &lt;think&gt;

    • or &lt;think&gt; + &lt;answer&gt;

3. Constructed Alpaca-style records:

    • instruction: system prompt + question + history so far

    • input: always ""

    • output: the next model action to predict

The final dataset used for SFT is: iterative_sft_data.json

It is registered in LLaMA-Factory under the dataset name: afm_iterative



**Training Configuration**

Training was performed on Google Colab with GPU A100.

Command used:

!llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset afm_iterative \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir saves/Qwen2.5-3B-Iterative-SFT \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --fp16


 **train results and metrics**

  epoch                    =         1.0
  total_flos               = 652458402GF
  train_loss               =      0.8949
  train_runtime            =  2:51:59.12
  train_samples_per_second =       2.567
  train_steps_per_second   =       0.321



**Folder “Qwen2.5-3B-Iterative-SFT” Contains**

1. adapter_model.safetensors — LoRA weights
2. adapter_config.json — LoRA configuration
3. tokenizer / special tokens files
4. chat_template.jinja
4. training logs & metadata



**How to Load This Checkpoint (for the RL stage)**

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./Qwen2.5-3B-Iterative-SFT"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

model = PeftModel.from_pretrained(model, adapter_path)



**Notes for the RL stage**

This checkpoint is not a merged full model; it is a LoRA adapter.

You should load it on top of the base model and then continue PPO/GRPO.


The model already learned:

• step-wise reasoning with &lt;think&gt;

• generating &lt;search&gt; calls when needed

• switching to &lt;answer&gt; when ready
