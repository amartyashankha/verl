# Detailed Documentation: `run_qwen2-32b_dapo.sh` Parameters

This script runs DAPO (Data-Augmented Policy Optimization) training for the Qwen2.5-32B model on mathematical reasoning tasks with tool usage capabilities. Below is a comprehensive explanation of all parameters organized by category.

## 🗂️ Data and Model Configuration

### Path Configuration
- **`HDFS_ROOT`** (default: `$PWD`): Root directory for HDFS/distributed storage. Used as the base path for checkpoints.
- **`DATA_ROOT`** (default: `$PWD`): Root directory for datasets. Used as the base path for all data files.

### Dataset Paths
- **`dapo_math_17k`** (`$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k`): Path to DAPO-Math-17k dataset containing 17,000 mathematical problems for training
- **`aime_2024`** (`$DATA_ROOT/dataset/Maxwell-Jia/AIME_2024`): Path to AIME 2024 dataset - American Invitational Mathematics Examination problems 
- **`aime_2025`** (`$DATA_ROOT/dataset/yentinglin/aime_2025`): Path to AIME 2025 dataset - Used for validation/testing
- **`train_files`**: List of training datasets (currently only DAPO-Math-17k)
- **`test_files`**: List of validation datasets (currently only AIME 2025)

### Model Configuration
- **`model_path`** (`$HDFS_ROOT/checkpoint/multiturn-sft-qwen-2.5-32b-instruct/global_step_372`): Path to the base model checkpoint - Using a multi-turn SFT-tuned Qwen-2.5-32B-Instruct model at step 372

### Tool Configuration
- **`tool_config_path`** (`recipe/retool/sandbox_fusion_tool_config.yaml`): Path to tool configuration YAML file that defines the code interpreter tool for mathematical problem solving. This enables the model to write and execute Python code to solve math problems.

## 📊 Experiment Tracking

- **`project_name`** (`wuxibin_retool`): WandB project name for experiment tracking
- **`experiment_name`** (`qwen2.5-32b_dapo`): Specific experiment identifier for this run
- **`default_local_dir`** (`$DATA_ROOT/checkpoint/$experiment_name`): Local directory for saving checkpoints and logs

## 🎯 Algorithm Parameters

### Core Algorithm Settings
- **`adv_estimator`** (`grpo`): Advantage estimation method using Generalized Reward Policy Optimization
- **`use_kl_in_reward`** (`False`): Whether to include KL divergence penalty in reward calculation
- **`kl_coef`** (`0.0`): Coefficient for KL penalty in reward (disabled when 0)
- **`use_kl_loss`** (`False`): Whether to add KL divergence as a separate loss term
- **`kl_loss_coef`** (`0.0`): Coefficient for KL loss term (disabled when 0)

### PPO Clipping Parameters
- **`clip_ratio_low`** (`0.2`): Lower bound for PPO clipping ratio - Prevents overly aggressive policy updates
- **`clip_ratio_high`** (`0.28`): Upper bound for PPO clipping ratio - Allows slightly more exploration than standard PPO
- **`clip_ratio_c`** (`10.0`): Clipping ratio for advantage normalization in the custom clipping mechanism

## 🔧 Training Configuration

### Sequence Length Limits
- **`max_turns`** (`8`): Maximum number of conversation turns - Allows multi-turn tool usage and reasoning
- **`max_prompt_length`** (`2048`): Maximum tokens allowed in the prompt
- **`max_response_length`** (`16384`): Maximum tokens allowed in the response - Large to accommodate code execution and detailed solutions

### Learning Rate
- **`actor_lr`** (`1e-6`): Learning rate for the actor model - Conservative rate suitable for fine-tuning large models

### Batch Size Configuration
- **`train_batch_size`** (`512`): Global batch size for training across all GPUs
- **`ppo_mini_batch_size`** (`64`): Mini-batch size for PPO gradient updates
- **`n_resp_per_prompt`** (`16`): Number of responses to generate per prompt during training rollouts
- **`n_resp_per_prompt_val`** (`30`): Number of responses to generate per prompt during validation

## ⚡ Performance and Hardware Configuration

### Parallelism Settings
- **`infer_tp`** (`4`): Tensor parallelism size for vLLM inference - Splits model across 4 GPUs
- **`train_sp`** (`8`): Sequence parallelism size for training using Ulysses algorithm
- **`offload`** (`True`): Enable parameter and optimizer offloading to CPU to save GPU memory

### Memory Configuration
- **`actor_max_token_len_per_gpu`**: Maximum tokens per GPU for actor model (calculated as `(max_prompt_length + max_response_length) * 1`)
- **`log_prob_max_token_len_per_gpu`**: Maximum tokens per GPU for log probability computation (4x actor tokens for reference model)

## 🚀 verl.trainer.main_ppo Parameters

### Data Processing
- **`data.return_raw_chat`** (`True`): Return raw chat format instead of tokenized data
- **`data.filter_overlong_prompts`** (`True`): Filter out prompts that exceed max length
- **`data.truncation`** (`'error'`): Error out if sequences exceed max length (instead of truncating)
- **`data.custom_cls.path`** (`recipe/retool/retool.py`): Path to custom dataset class
- **`data.custom_cls.name`** (`CustomRLHFDataset`): Custom dataset class that handles AIME datasets

### Reward Function
- **`custom_reward_function.path`** (`recipe/retool/retool.py`): Path to custom reward function
- **`custom_reward_function.name`** (`compute_score`): Function that scores mathematical solutions and encourages tool usage

### Model Configuration
- **`actor_rollout_ref.model.use_remove_padding`** (`True`): Remove padding tokens for efficiency
- **`actor_rollout_ref.model.enable_gradient_checkpointing`** (`True`): Enable gradient checkpointing to save memory

### Actor Training
- **`actor_rollout_ref.actor.use_dynamic_bsz`** (`True`): Dynamically adjust batch size based on sequence lengths
- **`actor_rollout_ref.actor.ppo_max_token_len_per_gpu`**: Maximum tokens per GPU for PPO training
- **`actor_rollout_ref.actor.ulysses_sequence_parallel_size`**: Size of sequence parallelism
- **`actor_rollout_ref.actor.fsdp_config.param_offload`**: Enable FSDP parameter offloading
- **`actor_rollout_ref.actor.fsdp_config.optimizer_offload`**: Enable FSDP optimizer offloading

### Reference Model
- **`actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`**: Maximum tokens for reference model log probabilities

### Rollout Configuration
- **`actor_rollout_ref.rollout.name`** (`vllm`): Use vLLM for efficient inference
- **`actor_rollout_ref.rollout.mode`** (`async`): Asynchronous rollout for better throughput
- **`actor_rollout_ref.rollout.tensor_model_parallel_size`**: TP size for vLLM
- **`actor_rollout_ref.rollout.gpu_memory_utilization`** (`0.9`): Use 90% of GPU memory for vLLM

### Multi-turn Configuration
- **`actor_rollout_ref.rollout.multi_turn.enable`** (`True`): Enable multi-turn conversations
- **`actor_rollout_ref.rollout.multi_turn.max_user_turns`**: Maximum user turns in conversation
- **`actor_rollout_ref.rollout.multi_turn.max_assistant_turns`**: Maximum assistant turns
- **`actor_rollout_ref.rollout.multi_turn.tool_config_path`**: Path to tool configuration
- **`actor_rollout_ref.rollout.multi_turn.format`** (`hermes`): Use Hermes format for tool calls

### Sampling Parameters
- **`actor_rollout_ref.rollout.val_kwargs.top_p`** (`0.6`): Top-p sampling for validation
- **`actor_rollout_ref.rollout.val_kwargs.temperature`** (`1.0`): Temperature for validation sampling
- **`actor_rollout_ref.rollout.val_kwargs.n`**: Number of samples for validation

## 📝 Trainer Configuration

### Logging
- **`trainer.logger`** (`['console','wandb']`): Enable both console and WandB logging
- **`trainer.project_name`**: WandB project name
- **`trainer.experiment_name`**: Experiment identifier

### Hardware Configuration
- **`trainer.n_gpus_per_node`** (`8`): Number of GPUs per node
- **`trainer.nnodes`** (`2`): Number of nodes for distributed training (16 total GPUs)

### Training Schedule
- **`trainer.val_before_train`** (`True`): Run validation before starting training
- **`trainer.log_val_generations`** (`100`): Log 100 validation generations to WandB
- **`trainer.save_freq`** (`30`): Save checkpoint every 30 steps
- **`trainer.test_freq`** (`5`): Run validation every 5 steps
- **`trainer.total_epochs`** (`1`): Train for 1 complete epoch
- **`trainer.default_local_dir`**: Directory for saving checkpoints

## 🛠️ Tool Configuration Details

The tool configuration file (`sandbox_fusion_tool_config.yaml`) defines:
- **Custom Sandbox Tool**: `CustomSandboxFusionTool` for code execution
- **Sandbox URL**: Endpoint for code execution service
- **Rate Limiting**: 128 concurrent requests with global rate limiting
- **Execution Limits**: 30s timeout, 1GB memory limit
- **Tool Schema**: OpenAI function calling format for code interpreter

## Summary

This configuration sets up a sophisticated DAPO training pipeline that:
1. Fine-tunes a 32B parameter model on mathematical reasoning
2. Uses multi-turn conversations with tool usage (code execution)
3. Employs distributed training across 16 GPUs (2 nodes × 8 GPUs)
4. Implements memory-efficient training with offloading and gradient checkpointing
5. Uses custom reward functions that encourage both correctness and tool usage
6. Validates on challenging AIME competition problems
