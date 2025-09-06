#!/bin/bash
set -x

export VLLM_USE_V1=1

# ================= data/model/config =================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

# Dataset paths - update these to your orchestrator datasets
train_files="['$DATA_ROOT/dataset/orchestrator/train.json']"
test_files="['$DATA_ROOT/dataset/orchestrator/test.json']"

# Model path - UPDATE THIS to your 14B model path
model_path=$DATA_ROOT/checkpoint/qwen-2.5-7b-instruct-sft

# Modal configuration
modal_base_url="https://fairies--incremental-leader-agent-api"
modal_timeout=300
modal_evaluation_url="https://fairies--swe-gym-evaluation-service-polling-fastapi-app.modal.run"

# Agent loop configuration
agent_loop_config_path=recipe/orchestrator/agent_loop_config.yaml

# wandb
project_name=orchestrator
experiment_name=qwen2.5-14b_orchestrator_modal
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
# PPO/GRPO settings
adv_estimator=gae  # or grpo
use_kl_in_reward=True
kl_coef=0.05
use_kl_loss=True
kl_loss_coef=0.1

# Training parameters
max_turns=20  # Allow more turns for complex notebook interactions
max_prompt_length=4096
max_response_length=32768  # Larger for notebook outputs

# Reduced learning rate for larger model stability
actor_lr=1e-7  # Reduced from 5e-7

# Reduced batch sizes for 14B model (approx 50% of 7B values)
train_batch_size=16  # Reduced from 32
ppo_mini_batch_size=4  # Reduced from 8
n_resp_per_prompt=4
n_resp_per_prompt_val=8

# Truncation settings
enable_truncation=true
truncation_strategy="ast_llm_compaction"  # or "first_user_priority"
truncation_max_tokens=16000

# ================= performance =================
# Increased parallelism for 14B model
infer_tp=8  # Increased from 4 - splits model across 8 GPUs for inference
train_sp=8  # Increased from 4 - sequence parallel for training

# Keep offloading enabled for memory efficiency
offload=True

# Adjusted token limits for 14B model (reduced by ~50%)
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) / 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/orchestrator/orchestrator.py \
    data.custom_cls.name=OrchestratorDataset \
    data.custom_cls.kwargs.enable_truncation=$enable_truncation \
    data.custom_cls.kwargs.truncation_strategy=$truncation_strategy \
    data.custom_cls.kwargs.truncation_max_tokens=$truncation_max_tokens \
    custom_reward_function.path=recipe/orchestrator/orchestrator.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.modal_base_url=$modal_base_url \
    actor_rollout_ref.rollout.multi_turn.modal_timeout=$modal_timeout \
    actor_rollout_ref.rollout.multi_turn.enable_truncation=$enable_truncation \
    actor_rollout_ref.rollout.multi_turn.truncation_strategy=$truncation_strategy \
    actor_rollout_ref.rollout.multi_turn.truncation_max_tokens=$truncation_max_tokens \
    actor_rollout_ref.rollout.multi_turn.format=default \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@