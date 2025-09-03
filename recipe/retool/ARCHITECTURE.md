# ReTool Modal Architecture

## Overview
Modal implementation for training Qwen2.5-7B-Instruct on tool-calling tasks using the ReTool dataset. Supports both SFT (Supervised Fine-Tuning) and DAPO (PPO with GRPO) training.

## Architecture Decisions

### 1. Three-Volume System
**Decision**: Separate volumes for data, models, and checkpoints

```
retool-sft-data/       # Training datasets
retool-sft-models/     # Base models
retool-sft-checkpoints/# Training outputs
```

**Rationale**: 
- Clean separation of concerns
- Independent lifecycle management
- Reusability across experiments

### 2. Unified Training Script
**Decision**: Single `retool_modal_sft.py` handles both SFT and DAPO

**Benefits**:
- Shared infrastructure (volumes, image, auth)
- Automatic checkpoint discovery
- Consistent patterns

### 3. Simplified Sandbox Service
**Decision**: Created lightweight code execution service instead of using original Docker container

**Original Design**: 
- Uses `volcengine/sandbox-fusion` Docker container
- Full sandboxed execution environment
- Supports multiple programming languages

**Our Implementation**:
```python
@modal.fastapi_endpoint()
def run_code(data: dict):
    # Simple subprocess execution
    subprocess.run(["python3", temp_file], ...)
```

**Rationale**:
- Quick implementation for development/testing
- Avoids Docker container complexity on Modal
- Sufficient for Python-only use case
- Note: Not production-secure like original

### 4. Direct Parameter Translation
**Decision**: Map shell script parameters directly to Python

**Example**:
```bash
# Shell script
--model.path /models/Qwen2.5-7B-Instruct

# Python translation
"--model.path", "/models/Qwen2.5-7B-Instruct"
```

**Benefits**:
- Easy verification
- No abstraction overhead
- CLI override support

## Key Implementation Patterns

### Image Building
```python
image = (
    modal.Image.from_registry("verlai/verl:latest")
    .env(secrets_dict)              # 1. Environment
    .pip_install(requirements)      # 2. Dependencies  
    .add_local_python_source(...)   # 3. MUST be last!
)
```

### Volume Management
```python
@app.function(volumes={"/data": data_volume})
def process():
    data_volume.reload()  # Always reload!
    # Use absolute paths
```

### Distributed Training
- **SFT**: Uses torchrun (PyTorch distributed)
- **DAPO**: Uses Ray (different initialization)
- Both need proper paths and environment setup

## Lessons Learned

### Critical Discoveries
1. **SFT checkpoints are already HuggingFace format** - No merge needed
2. **Path types matter** - verl wants file paths, not module paths
3. **Dataset names must be exact** - Check HuggingFace for correct names

### Modal-Specific Insights
1. **add_local_* must be last** - Modal optimization requirement
2. **Volume reload is mandatory** - Data may be stale otherwise
3. **Simple subprocess better than Docker** - For basic code execution needs

### Error Patterns
1. **WandB 403** → Check entity/permissions
2. **Import errors** → Mount recipe directory
3. **NCCL failures** → Verify GPU config
4. **OOM** → Reduce batch size

## File Organization
```
recipe/retool/
├── retool_modal_sft.py          # Main training script
├── modal_fusion_sandbox.py      # Sandbox service
├── test_sandbox_fusion.py       # Sandbox tests
├── sandbox_fusion_modal_config.yaml  # Tool config
├── env.example                  # Environment template
├── MODAL_GUIDE.md              # User guide
└── ARCHITECTURE.md             # This file

# Project root:
├── .env                        # API tokens (gitignored)
└── .env.example               # Environment template
```

## Future Improvements
1. **Multi-node support** - Extend beyond single node
2. **Pipeline automation** - Chain SFT → DAPO
3. **Cost optimization** - A100 support, spot instances
4. **Evaluation suite** - Automatic benchmarking
