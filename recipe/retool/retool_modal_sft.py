# ---
# cmd: ["modal", "run", "recipe/retool/retool_modal_sft.py::train"]
# ---

# # Train Qwen2.5-7B with SFT on ReTool dataset using Modal

# This script implements supervised fine-tuning (SFT) for multi-turn tool-calling
# conversations using the verl framework on Modal.

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Required environment variables with error handling
def get_required_env(var_name: str) -> str:
    """Get required environment variable or raise helpful error."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {var_name}\n"
            f"Please create a .env file in the project root with:\n"
            f"{var_name}=your_value_here"
        )
    return value


HF_TOKEN = get_required_env("HF_TOKEN")
WANDB_API_KEY = get_required_env("WANDB_API_KEY")

# ## Define the Modal app and constants

app = modal.App("retool-verl-training")

# Configuration constants
VERL_REPO_PATH = Path("/root/verl")
DATA_PATH = Path("/data")
MODEL_PATH = Path("/models")
CHECKPOINT_PATH = Path("/checkpoints")

# Optional environment variables with defaults
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "peteraltera-altera-al")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "retool-verl-modal")

# Training configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EXPERIMENT_NAME = "multiturn-sft-qwen-2.5-7b-instruct"

# Hardware configuration
DEFAULT_GPU_TYPE = "H100:8"
DEFAULT_N_GPUS = 8
DEFAULT_NNODES = 1

# Training hyperparameters
DEFAULT_MAX_LENGTH = 16384
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_MICRO_BATCH_SIZE_PER_GPU = 4
DEFAULT_TOTAL_EPOCHS = 6
DEFAULT_SAVE_FREQ = 62
DEFAULT_ULYSSES_PARALLEL_SIZE = 4

# Dataset constants
RETOOL_DATASET = "JoeYing/ReTool-SFT"
AIME_DATASET = "Maxwell-Jia/AIME_2024"  # Match original script

# Volume names
DATA_VOLUME = "retool-sft-data"
MODEL_VOLUME = "retool-sft-models"
CHECKPOINT_VOLUME = "retool-sft-checkpoints"

# File paths
RETOOL_DATA_PATH = "ReTool-SFT/data"
RETOOL_DATA_FILE = "train-00000-of-00001.parquet"
AIME_DATA_PATH = "dataset/Maxwell-Jia/AIME_2024"  # Match original script

# ## Create the image with verl and dependencies

# TODO: Verify the best base image version for SFT training
image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .pip_install("verl[vllm]==0.4.1", "datasets", "omegaconf", "pyyaml", "httpx", "python-dotenv")
    .env(
        {
            "HF_TOKEN": HF_TOKEN,
            "HUGGINGFACE_HUB_TOKEN": HF_TOKEN,
            "WANDB_API_KEY": WANDB_API_KEY,
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
        }
    )
    # IMPORTANT: add_local_* must be last in the chain (Modal requirement)
    .add_local_file(
        Path(__file__).parent / "sandbox_fusion_modal_config.yaml",
        remote_path="/root/sandbox_fusion_modal_config.yaml",
    )
    .add_local_python_source("recipe.retool")
    .add_local_dir(
        Path(__file__).parent.parent,
        remote_path=f"{VERL_REPO_PATH}/recipe",
    )
)

# ## Define volumes for persistent storage

data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)

# ## Data preparation function


@app.function(
    image=image,
    volumes={DATA_PATH: data_volume},
    timeout=60 * 60,  # 60 minutes for multiple datasets
)
def prep_dataset():
    """Download and preprocess all datasets for SFT and DAPO training."""
    import sys

    from huggingface_hub import snapshot_download

    print("Starting dataset preparation...")

    # 1. ReTool-SFT dataset (for SFT training)
    print("\n📥 Downloading ReTool-SFT dataset...")

    # Import and run the preprocessing script
    from recipe.retool import retool_sft_preprocess

    # Temporarily change sys.argv to simulate command line args
    original_argv = sys.argv
    sys.argv = ["retool_sft_preprocess.py"]

    # Override the save path in the preprocessing module
    import json

    import datasets

    # Define tool schema directly (from sandbox_fusion_tool_config.yaml)
    tool_schema = {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "A tool for executing code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "The code to execute."}},
                "required": ["code"],
            },
        },
    }
    tools = json.dumps([tool_schema])

    # Download and process dataset using the module's functions
    print(f"Downloading {RETOOL_DATASET} dataset...")
    data = datasets.load_dataset(RETOOL_DATASET)["train"]

    print("Processing dataset...")
    data = data.map(retool_sft_preprocess.process, fn_kwargs={"tools": tools})

    # Save to our volume path
    save_path = DATA_PATH / RETOOL_DATA_PATH
    save_path.mkdir(parents=True, exist_ok=True)
    output_file = save_path / RETOOL_DATA_FILE

    data.to_parquet(output_file)
    print(f"Dataset saved to {output_file}")

    # Log some statistics for debugging
    print(f"Total examples: {len(data)}")
    print(f"First example keys: {list(data[0].keys()) if len(data) > 0 else 'No data'}")
    print(f"Number of messages in first example: {len(data[0]['messages']) if len(data) > 0 else 0}")

    # Restore sys.argv
    sys.argv = original_argv

    # 2. DAPO-Math-17k dataset (for RL training)
    print("\n📥 Downloading DAPO-Math-17k dataset...")
    dapo_path = DATA_PATH / "dataset/BytedTsinghua-SIA/DAPO-Math-17k"

    try:
        snapshot_download(
            repo_id="BytedTsinghua-SIA/DAPO-Math-17k",
            repo_type="dataset",
            local_dir=dapo_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ DAPO-Math-17k dataset saved to {dapo_path}")
    except Exception as e:
        print(f"❌ Failed to download DAPO-Math-17k dataset: {e}")
        raise

    # 3. AIME-2024 dataset (for evaluation)
    print(f"\n📥 Downloading {AIME_DATASET} dataset...")
    aime_path = DATA_PATH / AIME_DATA_PATH

    try:
        snapshot_download(
            repo_id=AIME_DATASET,
            repo_type="dataset",
            local_dir=aime_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ {AIME_DATASET} dataset saved to {aime_path}")
    except Exception as e:
        print(f"❌ Failed to download {AIME_DATASET} dataset: {e}")
        raise

    # 4. AIME-2025 dataset (for evaluation)
    print("\n📥 Downloading yentinglin/aime_2025 dataset...")
    aime_2025_path = DATA_PATH / "dataset/yentinglin/aime_2025"

    try:
        snapshot_download(
            repo_id="yentinglin/aime_2025",
            repo_type="dataset",
            local_dir=aime_2025_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ AIME-2025 dataset saved to {aime_2025_path}")
    except Exception as e:
        print(f"❌ Failed to download AIME-2025 dataset: {e}")
        raise

    print("\n✅ All datasets downloaded successfully!")
    print(f"  - ReTool-SFT: {output_file}")
    print(f"  - DAPO-Math-17k: {dapo_path}")
    print(f"  - AIME-2024: {aime_path}")
    print(f"  - AIME-2025: {aime_2025_path}")


# ## Model download function


@app.function(
    image=image,
    volumes={MODEL_PATH: model_volume},
    timeout=60 * 60,  # 1 hour for large model download
)
def download_model(model_name: str = DEFAULT_MODEL):
    """Download the base model from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading model: {model_name}")

    local_dir = MODEL_PATH / model_name.replace("/", "_")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Model saved to {local_dir}")
    return str(local_dir)


# ## Training function


@app.function(
    image=image,
    gpu=DEFAULT_GPU_TYPE,
    volumes={
        DATA_PATH: data_volume,
        MODEL_PATH: model_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
    timeout=24 * 60 * 60,  # 24 hours
    # TODO: Configure NCCL settings for Ulysses sequence parallelism
    # env={"NCCL_DEBUG": "INFO"},
)
def train(*arglist):
    """Run SFT training on the ReTool dataset."""
    import argparse

    # Parse arguments with default experiment name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Experiment name for training",
        dest="experiment_name",  # Store as experiment_name internally
    )

    # Parse known args to allow passing through verl parameters
    known_args, unknown_args = parser.parse_known_args(args=arglist)
    experiment_name = known_args.experiment_name

    # Reload volumes to get latest data
    data_volume.reload()
    model_volume.reload()

    print(f"Starting SFT training: {experiment_name}")
    print(f"Additional verl args: {unknown_args}")

    # Define paths
    train_data = DATA_PATH / RETOOL_DATA_PATH / RETOOL_DATA_FILE
    eval_data = train_data  # Using same file for eval as in original script
    model_path = MODEL_PATH / DEFAULT_MODEL.replace("/", "_")
    save_path = CHECKPOINT_PATH / experiment_name

    # Build the training command using constants
    cmd = [
        "torchrun",
        f"--nnodes={DEFAULT_NNODES}",
        f"--nproc_per_node={DEFAULT_N_GPUS}",
        "--master_addr=localhost",
        "--master_port=29500",
        "--node_rank=0",
        "-m",
        "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={train_data}",
        f"data.val_files={eval_data}",
        f"data.max_length={DEFAULT_MAX_LENGTH}",
        f"data.train_batch_size={DEFAULT_TRAIN_BATCH_SIZE}",
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        "data.multiturn.tools_key=tools",
        f"data.micro_batch_size_per_gpu={DEFAULT_MICRO_BATCH_SIZE_PER_GPU}",
        f"model.partial_pretrain={model_path}",
        "model.strategy=fsdp",
        f"trainer.default_local_dir={save_path}",
        f"trainer.project_name={WANDB_PROJECT}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.total_epochs={DEFAULT_TOTAL_EPOCHS}",
        f"trainer.save_freq={DEFAULT_SAVE_FREQ}",
        f"ulysses_sequence_parallel_size={DEFAULT_ULYSSES_PARALLEL_SIZE}",
        "use_remove_padding=true",
        # TODO: Add any additional SFT-specific settings if needed
    ]

    # WandB is always enabled (environment variables are set in image)
    cmd.append("trainer.logger=['console','wandb']")
    print(f"📊 WandB logging enabled: {WANDB_ENTITY}/{WANDB_PROJECT}")

    # Add any additional arguments from CLI
    if arglist:
        cmd.extend(arglist)

    print("Training command:")
    print(" ".join(cmd))

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        raise

    # Log checkpoint info
    print(f"\n📁 Checkpoints saved to: {save_path}")
    print("To merge to HuggingFace format, run:")
    print(f"  modal run recipe/retool/retool_modal_sft.py::merge_checkpoint --experiment-name={experiment_name}")


# ## Checkpoint management functions


@app.function(
    image=image,
    gpu="A100",  # Need GPU for model loading and merging
    volumes={CHECKPOINT_PATH: checkpoint_volume},
    timeout=30 * 60,  # 30 minutes
)
def merge_checkpoint(*arglist):
    """Merge verl checkpoint to HuggingFace format for DAPO training."""
    import argparse
    import json
    import subprocess
    from datetime import datetime

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, help="Experiment name to merge")
    parser.add_argument("--step", type=int, help="Specific step to merge")
    args = parser.parse_args(arglist)

    experiment_name = args.experiment_name
    step = args.step

    print("🔀 Starting checkpoint merge to HuggingFace format...")

    # Organize checkpoints in subdirectories
    sft_dir = CHECKPOINT_PATH / "sft-experiments"

    # Find the experiment directory
    if experiment_name:
        exp_dir = sft_dir / experiment_name
        if not exp_dir.exists():
            # Try without sft-experiments prefix (legacy)
            exp_dir = CHECKPOINT_PATH / experiment_name
    else:
        # Find latest experiment
        exp_dirs = []
        for d in [sft_dir, CHECKPOINT_PATH]:
            if d.exists():
                exp_dirs.extend([x for x in d.iterdir() if x.is_dir()])

        if not exp_dirs:
            print("❌ No experiment directories found!")
            return None

        exp_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
        experiment_name = exp_dir.name
        print(f"📁 Using latest experiment: {experiment_name}")

    # Find checkpoint directory
    if step is not None:
        checkpoint_dir = exp_dir / f"global_step_{step}"
        if not checkpoint_dir.exists():
            print(f"❌ Checkpoint global_step_{step} not found!")
            return None
    else:
        # Find latest checkpoint
        checkpoint_dirs = sorted([d for d in exp_dir.iterdir() if d.name.startswith("global_step_") and d.is_dir()])

        if not checkpoint_dirs:
            print(f"❌ No checkpoints found in {experiment_name}!")
            return None

        checkpoint_dir = checkpoint_dirs[-1]
        step = int(checkpoint_dir.name.split("_")[-1])

        print(f"📁 Merging checkpoint: {checkpoint_dir.name}")

    # Debug: Check checkpoint structure
    print("\n🔍 Checkpoint directory contents:")
    try:
        import os

        for item in os.listdir(checkpoint_dir):
            item_path = checkpoint_dir / item
            if item_path.is_dir():
                print(f"  📁 {item}/")
                # Show contents of subdirectories
                for subitem in os.listdir(item_path):
                    print(f"    - {subitem}")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  ❌ Error listing contents: {e}")

    # Check if already merged
    hf_dir = checkpoint_dir / "huggingface"
    if hf_dir.exists():
        print(f"✅ Already merged to HuggingFace format at: {hf_dir}")
        return str(hf_dir)

    # Run merge command
    cmd = [
        "python",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint_dir),
        "--target_dir",
        str(hf_dir),
    ]

    print("Running merge command:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print("✅ Checkpoint merged successfully!")

        # Save metadata
        metadata = {
            "experiment_name": experiment_name,
            "global_step": step,
            "merged_at": datetime.now().isoformat(),
            "source_dir": str(checkpoint_dir),
            "hf_dir": str(hf_dir),
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "training_type": "sft",
        }

        with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"📍 HuggingFace checkpoint ready at: {hf_dir}")
        print("Use this path for DAPO training")

        return str(hf_dir)

    except subprocess.CalledProcessError as e:
        print(f"❌ Merge failed: {e}")
        raise


@app.function(
    image=image,
    volumes={CHECKPOINT_PATH: checkpoint_volume},
)
def list_checkpoints():
    """List all available checkpoints and their merge status."""

    print("📋 Available Checkpoints\n")

    # Check both legacy and new structure
    dirs_to_check = [
        (CHECKPOINT_PATH / "sft-experiments", "SFT"),
        (CHECKPOINT_PATH / "dapo-experiments", "DAPO"),
        (CHECKPOINT_PATH, "Legacy"),
    ]

    for base_dir, label in dirs_to_check:
        if not base_dir.exists():
            continue

        print(f"\n{label} Experiments:")
        print("-" * 50)

        for exp_dir in sorted(base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            print(f"\n📁 {exp_dir.name}")

            # Find checkpoints
            checkpoints = sorted([d for d in exp_dir.iterdir() if d.name.startswith("global_step_") and d.is_dir()])

            if not checkpoints:
                print("   No checkpoints found")
                continue

            for ckpt in checkpoints:
                hf_exists = (ckpt / "huggingface").exists()
                status = "✅ (HF ready)" if hf_exists else "⚠️  (verl only)"
                print(f"   - {ckpt.name} {status}")

                # Show metadata if available
                info_file = ckpt / "checkpoint_info.json"
                if info_file.exists():
                    import json

                    with open(info_file) as f:
                        info = json.load(f)
                    print(f"     Merged: {info.get('merged_at', 'N/A')}")


@app.function(
    image=image,
    volumes={CHECKPOINT_PATH: checkpoint_volume},
)
def get_latest_sft_checkpoint() -> Optional[str]:
    """Get the path to the latest merged SFT checkpoint ready for DAPO."""

    # Look for SFT experiments
    sft_dir = CHECKPOINT_PATH / "sft-experiments"
    if not sft_dir.exists():
        # Check legacy location
        sft_dir = CHECKPOINT_PATH

    # Find all HF checkpoints
    hf_checkpoints = []
    for exp_dir in sft_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        for ckpt_dir in exp_dir.iterdir():
            if ckpt_dir.name.startswith("global_step_"):
                hf_dir = ckpt_dir / "huggingface"
                if hf_dir.exists():
                    hf_checkpoints.append((hf_dir, ckpt_dir.stat().st_mtime))

    if not hf_checkpoints:
        print("❌ No merged HuggingFace checkpoints found!")
        print("Run merge_checkpoint first")
        return None

    # Return most recent
    latest = max(hf_checkpoints, key=lambda x: x[1])[0]
    print(f"📍 Latest SFT checkpoint: {latest}")
    return str(latest)


# ## DAPO Training with Tool Execution


@app.function(
    image=image,
    gpu="H100:8",  # 8 GPUs for DAPO training
    volumes={
        DATA_PATH: data_volume,
        MODEL_PATH: model_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
    timeout=24 * 60 * 60,  # 24 hours for RL training
)
def train_dapo(*arglist):
    """Run DAPO (PPO with GRPO) training with tool execution."""
    import argparse
    import os
    import subprocess

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        default="qwen2.5-7b-dapo",
        help="Experiment name for DAPO training",
        dest="experiment_name",
    )
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Path to SFT checkpoint (auto-detects if not specified)",
        dest="sft_checkpoint",
    )

    # Parse known args to allow passing through verl parameters
    known_args, unknown_args = parser.parse_known_args(args=arglist)
    experiment_name = known_args.experiment_name
    sft_checkpoint = known_args.sft_checkpoint

    # Reload volumes
    data_volume.reload()
    model_volume.reload()
    checkpoint_volume.reload()

    print(f"🚀 Starting DAPO training: {experiment_name}")
    print(f"Additional verl args: {unknown_args}")

    # Auto-detect SFT checkpoint if not specified
    if sft_checkpoint is None:
        # Look for latest SFT checkpoint (not merged)
        print("🔍 Auto-detecting latest SFT checkpoint...")

        # Look for SFT experiments
        sft_dir = CHECKPOINT_PATH / "sft-experiments"
        if not sft_dir.exists():
            # Check legacy location
            sft_dir = CHECKPOINT_PATH

        # Find all checkpoints
        checkpoints = []
        for exp_dir in sft_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            for ckpt_dir in exp_dir.iterdir():
                if ckpt_dir.name.startswith("global_step_"):
                    # Check if it has model files (safetensors or bin files)
                    has_model = any(
                        f.name.endswith((".safetensors", ".bin", "config.json"))
                        for f in ckpt_dir.iterdir()
                        if f.is_file()
                    )
                    if has_model:
                        checkpoints.append((ckpt_dir, ckpt_dir.stat().st_mtime))

        if not checkpoints:
            print("❌ No SFT checkpoints found!")
            print("Run SFT training first")
            return

        # Get most recent
        latest = max(checkpoints, key=lambda x: x[1])[0]
        sft_checkpoint = str(latest)
        print(f"✅ Auto-detected SFT checkpoint: {sft_checkpoint}")

    print(f"📍 Using SFT checkpoint: {sft_checkpoint}")

    # Dataset paths
    dapo_data = DATA_PATH / "dataset/BytedTsinghua-SIA/DAPO-Math-17k"
    aime_2024 = DATA_PATH / "dataset/Maxwell-Jia/AIME_2024"
    aime_2025 = DATA_PATH / "dataset/yentinglin/aime_2025"

    # Training output path
    save_path = CHECKPOINT_PATH / "dapo-experiments" / experiment_name

    # Load sandbox URL from config
    sandbox_config_path = "/root/sandbox_fusion_modal_config.yaml"

    # Change to verl directory where recipe/retool/retool.py exists
    os.chdir(VERL_REPO_PATH)
    print(f"📍 Working directory: {os.getcwd()}")

    # Build DAPO training command (based on run_qwen2_7b_dapo.sh)
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        # Algorithm settings
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.0",
        # Data settings
        f"data.train_files=['{dapo_data}']",
        f"data.val_files=['{aime_2025}', '{aime_2024}']",
        "data.return_raw_chat=True",
        "data.train_batch_size=64",
        "data.max_prompt_length=2048",
        "data.max_response_length=16384",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.custom_cls.path=recipe/retool/retool.py",
        "data.custom_cls.name=CustomRLHFDataset",
        # Custom reward function
        "custom_reward_function.path=recipe/retool/retool.py",
        "custom_reward_function.name=compute_score",
        # Model settings
        f"actor_rollout_ref.model.path={sft_checkpoint}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # Actor settings
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480",  # (2048+16384)*1
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=4",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        # Reference model settings
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=81920",  # 20480*4
        # Rollout settings
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
        # Multi-turn settings with tool execution
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=16",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={sandbox_config_path}",
        "actor_rollout_ref.rollout.multi_turn.format=hermes",
        # Generation settings
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.9",
        "actor_rollout_ref.rollout.n=16",  # n responses per prompt
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.6",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.n=30",  # n responses for validation
        # Trainer settings
        "trainer.logger=['console','wandb']",
        f"trainer.project_name={WANDB_PROJECT}",
        f"trainer.experiment_name={experiment_name}",
        "trainer.n_gpus_per_node=8",
        "trainer.val_before_train=True",
        "trainer.log_val_generations=20",
        "trainer.nnodes=1",
        "trainer.save_freq=20",
        f"trainer.default_local_dir={save_path}",
        "trainer.test_freq=10",
        "trainer.total_epochs=1",
    ]

    # Add any additional arguments from CLI
    if unknown_args:
        cmd.extend(unknown_args)

    # Set environment variables
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"  # Get full error traces

    print("DAPO training command:")
    print(" ".join(cmd))

    # Run training without capturing output to see real-time logs
    try:
        # Use subprocess.run without capture_output to see logs in real-time
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"❌ DAPO training failed with exit code: {result.returncode}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        print("✅ DAPO training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ DAPO training failed with exit code: {e.returncode}")
        raise

    print(f"\n📁 DAPO checkpoints saved to: {save_path}")


@app.function(
    image=image,
    timeout=5 * 60,  # 5 minutes
)
def check_sandbox_connection():
    """Test connection to Sandbox Fusion service."""
    import httpx
    import yaml

    print("🔍 Checking Sandbox Fusion connection...")

    # Load config
    config_path = "/root/sandbox_fusion_modal_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sandbox_url = config["tools"][0]["config"]["sandbox_fusion_url"]
    print(f"📍 Sandbox URL: {sandbox_url}")

    # Test connection by directly testing code execution
    try:
        # Test code execution (the sandbox accepts POST to root URL)
        test_code = "print('Hello from DAPO training!')"
        response = httpx.post(
            sandbox_url,
            json={"code": test_code, "language": "python"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "Success":
            print("✅ Sandbox service is reachable and working")
            print(f"   Output: {result['run_result']['stdout'].strip()}")
            return True
        else:
            print("❌ Code execution failed")
            print(f"   Response: {result}")
            return False

    except Exception as e:
        print(f"❌ Sandbox connection failed: {e}")
        print("Make sure to deploy the sandbox service first:")
        print("  modal deploy recipe/retool/modal_fusion_sandbox.py")
        raise


@app.local_entrypoint()
def main():
    """Example of running the full pipeline."""
    print("ReTool SFT Training Pipeline")
    print("1. Preparing dataset...")
    # prep_dataset.remote()

    print("2. Downloading model...")
    # download_model.remote()

    print("3. Starting training...")
    # train.remote()

    print("\nTo run individual steps:")
    print("  # Data and model preparation")
    print("  modal run retool_modal_sft.py::prep_dataset")
    print("  modal run retool_modal_sft.py::download_model")
    print("  \n  # SFT training")
    print("  modal run retool_modal_sft.py::train")
    print("  modal run retool_modal_sft.py::merge_checkpoint")
    print("  \n  # DAPO training")
    print("  modal run retool_modal_sft.py::check_sandbox_connection")
    print("  modal run retool_modal_sft.py::train_dapo")
    print("\nTo run training with custom args:")
    print("  modal run retool_modal_sft.py::train -- --experiment-name=my-experiment")
    print("  modal run retool_modal_sft.py::train_dapo -- trainer.total_epochs=2")
    print("\nTo debug WandB access:")
    print("  modal run retool_modal_sft.py::debug_wandb")


# ## Configuration Notes

# All configuration is centralized at the top of the file:
# - WandB settings: WANDB_ENTITY, WANDB_PROJECT
# - Model settings: DEFAULT_MODEL, DEFAULT_EXPERIMENT_NAME
# - Hardware: DEFAULT_GPU_TYPE, DEFAULT_N_GPUS, DEFAULT_NNODES
# - Training hyperparameters: batch sizes, epochs, etc.

# Environment variables are set in the image for all functions:
# - HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT

# To modify configuration:
# 1. Update constants at top of file for permanent changes
# 2. Pass CLI args for one-off overrides: trainer.total_epochs=10
# 3. Update secrets.py for authentication tokens

# ## Next Steps in the Full Pipeline

# After SFT training completes, the pipeline continues with:

# 1. Checkpoint Merging (TODO: add Modal function)
#    python -m verl.model_merger merge --backend fsdp \
#      --local_dir /checkpoints/xxx/global_step_168 \
#      --target_dir /checkpoints/xxx/global_step_168/huggingface

# 2. Sandbox Fusion Deployment (for code execution)
#    docker run -p 8080:8080 volcengine/sandbox-fusion:server-20250609
#    Consider: Modal service vs local deployment

# 3. DAPO Training (uses SFT checkpoint + both datasets)
#    - Loads HF checkpoint from step 1
#    - Uses ReTool-SFT + AIME-2024 datasets
#    - Connects to Sandbox Fusion for tool execution
