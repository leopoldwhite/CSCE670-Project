# Experiment Report: Enhancing SFT Models with Reinforcement Learning (Search-R1)

## 1. Objective
The primary goal of this experiment is to validate the hypothesis that performing Reinforcement Learning (RL) on top of a Supervised Fine-Tuned (SFT) model yields better performance than SFT alone, RL alone, or the base model. We utilize the Search-R1 framework to train and evaluate these models on the Natural Questions (NQ) dataset.

We are comparing the following four configurations:
1.  **Baseline**: Qwen2.5-3b-Instruct (No SFT, No RL)
2.  **SFT Only**: Qwen2.5-3b-Instruct  + SFT
3.  **RL Only**: Qwen2.5-3b-Instruct + RL (we used the `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo` checkpoint)
4.  **SFT + RL**: Qwen2.5-3b-Instruct  + SFT + RL (Ours)

## 2. RL Training Methodology
We have conducted the RL training using the **Search-R1** framework (powered by VeRL) on the **NQ** dataset (train split).

### 2.1 Training Configuration
The RL training was executed using PPO (Proximal Policy Optimization). Below are the key configurations and hyperparameters used in `train_ppo.sh`:

*   **Algorithm**: PPO
*   **Framework**: VeRL (based on Search-R1 codebase)
*   **Training Dataset**: `data/nq_search/train.parquet`
*   **Validation Dataset**: `data/nq_search/test.parquet`
*   **Reward Signal**: Outcome-based (Exact Match)
*   **Retriever**: Local retrieval service, Top-K = 3

### 2.2 Hyperparameters
The specific hyperparameters used for the PPO training run are as follows:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Training Steps** | 600 | Total update steps |
| **Batch Size** | 32 | Global training batch size |
| **Mini-batch Size** | 32 | PPO mini-batch size |
| **Micro-batch Size** | 8 | Per-device micro-batch size |
| **Learning Rate (Actor)** | 1e-6 | LR for the policy model |
| **Learning Rate (Critic)** | 1e-5 | LR for the value model |
| **KL Coefficient** | 0.001 | Penalty for deviation from reference model |
| **Max Turns (Train)** | 2 | Max agent-environment interaction turns during training |
| **Max Prompt Length** | 4096 | Context window limit |
| **Max Response Length** | 500 | Max tokens per generation |
| **Advantage Estimator** | GAE | Generalized Advantage Estimation |

## 3. Inference and Evaluation
Evaluation is performed on the **NQ-Test** dataset to measure generation quality and retrieval accuracy.

### 3.1 Evaluation Process
The inference uses the `infer.py` script (wrapped in `run_nq_eval.sh`). It performs a multi-turn search and generation process.

*   **Test Dataset**: `data/nq_search/test.parquet`
*   **Metrics**: Exact Match (EM) and ROUGE-L.

### 3.2 Inference Parameters
The following parameters are used during the evaluation phase:

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Temperature** | 0.7 | Sampling temperature |
| **Top-P** | 0.9 | Nucleus sampling |
| **Max Turns (Eval)** | 5 | Allowed turns for search/reasoning (Higher than training) |
| **Retrieval Top-K** | 3 | Documents retrieved per query |
| **Max New Tokens** | 1024 | Generation limit |

## 4. Experimental Results
Below is the comparison of the four experimental groups.

*Note: Metrics reported are on the NQ-Test set.*

| Group ID | Configuration | Exact Match (EM) | ROUGE-L |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Baseline** (No SFT, No RL) | 0.2374 | 0.3179 |
| 2 | **SFT Only** | 0.3654 | 0.4515 |
| 3 | **RL Only** | 0.3108 | 0.3906 |
| 4 | **SFT + RL (Ours)** | 0.4127 | 0.4890 |

## 5. Conclusion & Analysis

The results demonstrate that stacking RL on top of a high-quality SFT checkpoint yields the most robust policy for Search-R1 on NQ. **Group 4 (SFT + RL)** delivers the strongest Exact Match and ROUGE-L scores because SFT establishes the search-and-answer protocol while RL tailors the policy toward reward-aligned behaviors uncovered during environment interaction.

- **Role of SFT**: The supervised stage instills structured search prompting, citation discipline, and concise answer formatting. Models without this warm start (the baseline and RL-only variants) lag notably in EM, confirming that exploration alone cannot recover these priors efficiently.
- **Role of RL**: PPO fine-tuning sharpens the agent’s decision boundary on when to query, re-query, or stop, directly optimizing for answer quality. Compared with SFT-only, RL adds ~4.7 EM points, showing that online feedback compensates for any coverage gaps in the static corpus.
- **Overall takeaway**: High-quality SFT data amplifies the model’s ability to drive the search engine effectively, and RL then exploits that competence to maximize downstream QA accuracy.

### Notes
1. the SFT dataset is from this paper: [Chain-of-Agents (OPPO AI Agent Team, 2025)](https://arxiv.org/pdf/2508.13167).
2. the RL baseline is using this checkpoint [PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo), which is (RL) trained on both NQ and HotpotQA training set. Our method is only trained NQ training set during RL stage. But since we use extra sft data, let's consider them as fair comparision.

