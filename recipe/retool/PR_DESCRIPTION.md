# Add Modal-based Training Pipeline for ReTool Dataset

## Summary
This PR adds a Modal cloud implementation for training Qwen2.5-7B on the ReTool dataset, supporting both SFT (Supervised Fine-Tuning) and DAPO (Direct Alignment from Policy Optimization) training pipelines.

## Changes

### Core Implementation (`retool_modal_sft.py`)
- **Unified training script** handling both SFT and DAPO workflows
- **SFT Training**: Multi-turn tool-calling conversations with 8x H100 GPUs using Ulysses sequence parallelism
- **DAPO Training**: PPO with GRPO estimator, integrating code execution for mathematical problem solving
- **Volume management**: Three separate Modal volumes for data, models, and checkpoints
- **Auto-detection**: DAPO automatically finds and uses latest SFT checkpoint
- **Environment handling**: Secure API token management via `.env` files

### Key Features
- **Data preparation**: Downloads and prepares ReTool-SFT, AIME-2024, and DAPO-Math-17k datasets
- **Model management**: Downloads Qwen2.5-7B-Instruct from HuggingFace
- **Checkpoint handling**: Direct HuggingFace format support (no merge needed for DAPO)
- **Distributed training**: Uses torchrun for SFT and Ray for DAPO
- **WandB integration**: Full experiment tracking and logging

### Configuration Files
- **`sandbox_fusion_modal_config.yaml`**: Defines code interpreter tool configuration for DAPO
- **`.env.example`**: Template for required environment variables (HF_TOKEN, WANDB_API_KEY)

### Testing (`test_sandbox_fusion.py`)
- Comprehensive test suite for sandbox service
- Tests basic code execution, error handling, timeouts
- Includes single request testing utility

## Technical Details

### Training Parameters
- **SFT**: 2 epochs, batch size 32, max length 16384, gradient checkpointing
- **DAPO**: GRPO advantage estimation, 16 responses per prompt, multi-turn support up to 16 turns
- **Hardware**: 8x H100 GPUs with tensor/sequence parallelism

### Security Improvements
- Removed hardcoded API tokens from `secrets.py`
- All credentials now managed via environment variables
- Uses python-dotenv for secure token loading

## Usage

```bash
# Setup environment
cp recipe/retool/.env.example .env
# Edit .env with your tokens

# Run complete pipeline
modal run recipe/retool/retool_modal_sft.py::prep_dataset
modal run recipe/retool/retool_modal_sft.py::download_model
modal run recipe/retool/retool_modal_sft.py::train
modal run recipe/retool/retool_modal_sft.py::train_dapo
```

## Note
The sandbox service implementation (`modal_fusion_sandbox.py`) referenced in the code is intended to be a simplified subprocess-based code executor for development use, replacing the original Docker-based `volcengine/sandbox-fusion` container.
