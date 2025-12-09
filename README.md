# Enhancing Search-Tool Policies for Small LLMs via Process Distillation and Post-Distillation RL

> **Course Project Report**  
> **Authors**: Yuyang Bai, Shuning Gu, Zhuofeng Li, Yi Wen

This project explores whether small LLMs (e.g., 3B parameters) can learn effective search-tool policies by first acquiring structured reasoning traces and only then refining the policy with reinforcement learning. We build upon the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) framework for the reinforcement learning stage.

## Introduction

Large language models (LLMs) increasingly rely on external tools such as search engines to answer knowledge-intensive questions. For small LLMs (e.g., $\approx$ 3B parameters), learning when to call a search tool and whether search is needed is particularly challenging due to their limited parametric knowledge and weaker planning ability.

We propose a SFT $\to$ RL pipeline:
1.  **Process Distillation (SFT)**: We curate teacher-generated reasoning trajectories that explicitly annotate `<think>`, `<search>`, `<information>`, and `<answer>` actions. We perform process-supervised fine-tuning (SFT) to distill these iterative reasoning behaviors into the student model.
2.  **Post-Distillation RL**: We apply reinforcement learning using PPO (via the Search-R1 framework) to refine the agent’s decision boundaries for search timing, stopping conditions, and answer production.

## Methodology

### 1. Supervised Fine-tuning (SFT)
The goal of the SFT stage is to teach the student model to perform **iterative reasoning and search actions**.
*   **Model**: `Qwen2.5-3B-Instruct`
*   **Data**: Multi-step trajectories with `<think>`, `<search>`, `<information>`, and `<answer>` tags. The dataset is derived from [Chain-of-Agents (OPPO AI Agent Team, 2025)](https://arxiv.org/pdf/2508.13167).
*   **Training**: LoRA fine-tuning using `LLaMA-Factory`.

### 2. Reinforcement Learning (RL)
We use the **Search-R1** framework built on VeRL on the Natural Questions (NQ) dataset (train split).

*   **Objective**: Optimize the decision policy (when to search, how long, when to stop).
*   **Algorithm**: PPO (Proximal Policy Optimization).
*   **Reward Signal**: Outcome-based (Exact Match).
*   **Retriever**: Local retrieval service, Top-K = 3.

#### Hyperparameters
The specific hyperparameters used for the PPO training run in `train_ppo.sh` are:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Training Steps** | 600 | Total update steps |
| **Batch Size** | 32 | Global training batch size |
| **Mini-batch Size** | 32 | PPO mini-batch size |
| **Learning Rate (Actor)** | 1e-6 | LR for the policy model |
| **Learning Rate (Critic)** | 1e-5 | LR for the value model |
| **KL Coefficient** | 0.001 | Penalty for deviation from reference model |
| **Max Turns (Train)** | 2 | Max agent-environment interaction turns during training |
| **Max Prompt Length** | 4096 | Context window limit |
| **Max Response Length** | 500 | Max tokens per generation |

## Evaluation and Results

We evaluated our approach on the **NQ-Test** split using the `infer.py` script.

### Evaluation Parameters

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Temperature** | 0.7 | Sampling temperature |
| **Top-P** | 0.9 | Nucleus sampling |
| **Max Turns (Eval)** | 5 | Allowed turns for search/reasoning (Higher than training) |
| **Retrieval Top-K** | 3 | Documents retrieved per query |
| **Max New Tokens** | 1024 | Generation limit |

### Quantitative Results

| Group | Configuration | EM | ROUGE-L |
| :--- | :--- | :--- | :--- |
| 1 | **Baseline** (No SFT, No RL) | 0.2374 | 0.3179 |
| 2 | **SFT Only** | 0.3654 | 0.4515 |
| 3 | **RL Only*** | 0.3108 | 0.3906 |
| 4 | **SFT + RL (Ours)** | **0.4127** | **0.4890** |

**\*Note on RL Only**: This baseline uses the public checkpoint [`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo`](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo), which was trained on both NQ and HotpotQA. Our method (Group 4) was trained only on the NQ training set during the RL stage, making our strong performance even more significant despite the narrower RL data scope.

### Analysis
*   **Role of SFT**: SFT establishes the search-and-answer protocol. Models without this warm start (Baseline and RL Only) lag notably in EM, confirming that exploration alone cannot recover these priors efficiently.
*   **Role of RL**: Stacking RL on top of SFT yields the most robust policy. RL fine-tuning sharpens the agent’s decision boundary on when to query or stop, adding **~4.7 EM points** over SFT alone.

## Installation

This project uses the Search-R1 environment.

```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 

# install verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Retriever environment (Optional)
If you need to run the local retriever:

```bash
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

## Quick Start (Reproduction)

1.  **Launch Retriever**:
```bash
conda activate retriever
bash retrieval_launch.sh
```

2.  **Run RL Training**:
    Ensure you have the SFT checkpoint ready.
```bash
conda activate searchr1
bash train_ppo.sh
```

3.  **Run Evaluation**:
```bash
conda activate searchr1
    bash run_nq_eval.sh
    ```

## Acknowledgements

This project is built upon [Search-R1](https://github.com/PeterGriffinJin/Search-R1). We thank the authors for open-sourcing their framework.

**Original Search-R1 Citations**:
```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```
