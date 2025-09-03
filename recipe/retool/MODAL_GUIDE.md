# ReTool Modal Training Guide

## Overview

This guide documents the complete Modal implementation for training Qwen2.5-7B-Instruct on the ReTool dataset, including SFT (Supervised Fine-Tuning) and DAPO (Direct Alignment from Policy Optimization) training pipelines.

## What We Built

### 1. Complete Training Pipeline
- **SFT Training**: Multi-turn tool-calling conversations with code execution
- **DAPO Training**: PPO with GRPO estimator for tool-use optimization  
- **Sandbox Service**: Code execution environment for training
- **Distributed Training**: 8x H100 GPUs with tensor/sequence parallelism

### 2. Key Components
- `retool_modal_sft.py` - Main training script with SFT and DAPO
- `modal_fusion_sandbox.py` - Simplified code execution service (replaces Docker-based sandbox)
- `test_sandbox_fusion.py` - Sandbox test suite
- `sandbox_fusion_modal_config.yaml` - Tool configuration

## Quick Start

### Prerequisites
```bash
# Install Modal CLI
pip install modal
modal setup

# Configure environment variables
cp recipe/retool/env.example .env
# Edit .env with your actual tokens:
# HF_TOKEN=your_huggingface_token
# WANDB_API_KEY=your_wandb_api_key
```

### Run Complete Pipeline

```bash
# 1. Prepare datasets (ReTool-SFT, AIME-2024, DAPO-Math-17k)
modal run recipe/retool/retool_modal_sft.py::prep_dataset

# 2. Download Qwen2.5-7B-Instruct model
modal run recipe/retool/retool_modal_sft.py::download_model

# 3. Run SFT training
modal run recipe/retool/retool_modal_sft.py::train

# 4. Deploy sandbox (if not already deployed)
modal deploy recipe/retool/modal_fusion_sandbox.py

# 5. Run DAPO training (auto-detects latest SFT checkpoint)
modal run recipe/retool/retool_modal_sft.py::train_dapo
```

## Architecture & Design Decisions

### Volume Structure
Three separate volumes for clean separation:
- `retool-sft-data` - Training datasets
- `retool-sft-models` - Base models  
- `retool-sft-checkpoints` - Training outputs

**Rationale**: Independent lifecycle management, reusability across experiments

### Checkpoint Organization
```
retool-sft-checkpoints/
├── sft-experiments/
│   └── multiturn-sft-qwen-2.5-7b-instruct/
│       └── global_step_372/
│           └── actor/              # Direct checkpoint format
├── dapo-experiments/
│   └── multiturn-dapo-qwen-2.5-7b-instruct/
│       └── global_step_186/
└── checkpoint_info.json           # Metadata tracking
```

### Key Insights
1. **SFT checkpoints are already in HuggingFace format** - No merge needed for DAPO
2. **DAPO loads directly from SFT checkpoint** - Simplified pipeline
3. **Simplified sandbox implementation** - Basic subprocess execution replaces Docker container for development use

## Training Parameters

### SFT Training
- 8x H100 GPUs with Ulysses sequence parallelism
- 2 epochs on ReTool-SFT dataset
- Gradient checkpointing enabled
- Mixed precision (bf16)

### DAPO Training  
- 8x H100 GPUs with tensor + sequence parallelism
- PPO with GRPO baseline
- Multi-turn conversations (up to 16 turns)
- Integrates with sandbox for code execution

## Advanced Usage

### Custom Parameters
```bash
# SFT with custom settings
modal run recipe/retool/retool_modal_sft.py::train -- \
  --experiment-name=my-experiment \
  trainer.total_epochs=5 \
  trainer.save_freq=100

# DAPO with custom settings
modal run recipe/retool/retool_modal_sft.py::train_dapo -- \
  --experiment-name=my-dapo \
  trainer.total_epochs=3 \
  actor_rollout_ref.rollout.n=16
```

### Checkpoint Management
```bash
# List all checkpoints
modal run recipe/retool/retool_modal_sft.py::list_checkpoints

# Optional: Merge specific checkpoint to HF format
modal run recipe/retool/retool_modal_sft.py::merge_checkpoint \
  --experiment-name=multiturn-sft-qwen-2.5-7b-instruct \
  --step=372
```

### Debugging Tools
```bash
# Test sandbox connection
modal run recipe/retool/retool_modal_sft.py::check_sandbox_connection

# Test single code execution
modal run recipe/retool/test_sandbox_fusion.py::test_single_request \
  --code 'print(2+2)'

# View logs
modal logs -f --function-name train

# Check volumes
modal volume ls retool-sft-checkpoints
```

## Critical Learnings

### Modal-Specific Patterns

1. **Image Building Order Matters**
   ```python
   image = (
       modal.Image.from_registry("base")
       .env({"KEY": "value"})      # Environment vars
       .pip_install(...)           # Dependencies
       .add_local_python_source()  # MUST be last!
   )
   ```

2. **Sandbox Service Implementation**
   - Original uses `volcengine/sandbox-fusion` Docker container
   - We created simplified subprocess-based version for Modal
   - Sufficient for development, not production-secure

3. **Volume Usage**
   - Always call `volume.reload()` before reading
   - Use absolute paths for clarity
   - Volumes persist across function calls

### Training-Specific Insights

1. **SFT vs PPO/DAPO Differences**
   - SFT: Uses PyTorch distributed → needs `torchrun`
   - PPO/DAPO: Uses Ray → different initialization
   - Both need proper master/worker setup for multi-node

2. **Path Resolution**
   - verl expects file paths: `recipe/retool/retool.py`
   - Not module paths: `recipe.retool.retool`
   - Always `chdir` to repo root for consistency

3. **Dataset Names Matter**
   - Correct: `Maxwell-Jia/AIME_2024`
   - Wrong: `BytedTsinghua-SIA/AIME-2024`
   - Verify dataset names on HuggingFace

### Common Issues & Solutions

1. **WandB 403 Errors**
   - Set `WANDB_ENTITY` to your team/username
   - Or disable with `WANDB_MODE=offline`

2. **OOM Errors**
   - Reduce `data.micro_batch_size_per_gpu`
   - Enable gradient checkpointing (already on)
   - Consider fewer GPUs for testing

3. **NCCL Errors**
   - Add `NCCL_DEBUG=INFO` to environment
   - Check Ulysses sequence parallel config
   - Verify GPU interconnect

4. **Import Errors in Workers**
   - Mount recipe directory in volume
   - Use absolute imports
   - Ensure paths match between master/workers

## Cost Optimization

### Development
- Use 2-4 GPUs for testing: `trainer.n_gpus_per_node=2`
- Short runs: `trainer.total_epochs=1`
- Small batches: `data.train_batch_size=16`

### Production
- Use spot instances when available
- Consider A100s for cost/performance
- Multi-node for very large models

## Next Steps

1. **Evaluation Pipeline**: Add automatic evaluation on benchmarks
2. **Multi-Node Support**: Extend to distributed training across nodes
3. **Pipeline Automation**: Chain SFT → DAPO automatically
4. **Model Deployment**: Export and serve trained models

## Summary

This implementation provides a complete pipeline for training tool-calling models on Modal. The key features are:

- **Unified Script**: Single file handles both SFT and DAPO training
- **Automatic Checkpoint Detection**: DAPO finds and uses latest SFT checkpoint
- **Simplified Code Execution**: Basic sandbox service for development use
- **Clean Architecture**: Separation of concerns with multiple volumes and clear organization

**Note on Sandbox**: The original VERL uses the `volcengine/sandbox-fusion` Docker container for secure, multi-language code execution. Our Modal implementation uses a simplified subprocess-based approach suitable for development but not production security requirements.

The pipeline demonstrates how to adapt complex distributed training workflows to Modal's infrastructure.
