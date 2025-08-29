# Orchestrator Modal Notebook Agent Recipe

This recipe implements a notebook-style coding agent that executes code in Modal sandboxes with persistent state.

## Key Features

- **Stateful Execution**: Code blocks are executed sequentially in a persistent notebook environment
- **Modal Integration**: Uses Modal's sandbox API for secure code execution
- **AST-based Truncation**: Intelligent conversation compaction for long interactions
- **Multi-turn Support**: Handles complex back-and-forth interactions with code execution

## Architecture

The orchestrator agent differs from the tool agent in several ways:

1. **Code Extraction**: Uses `<code>...</code>` blocks instead of tool calls
2. **Stateful Environment**: Each agent maintains a persistent notebook throughout the conversation
3. **Sequential Execution**: Code blocks are executed in order, maintaining state between them
4. **Truncation Support**: Built-in support for AST-based conversation compaction

## Configuration

### Model and Tokenizer

The model is specified in the shell script:
```bash
model_path=$HDFS_ROOT/checkpoint/qwen-2.5-7b-instruct
```

You can use any Hugging Face compatible model. The tokenizer is automatically loaded from the same path.

### Modal Configuration

Configure the Modal API endpoint in the shell script:
```bash
modal_base_url="https://fairies--incremental-leader-agent-api"
modal_timeout=300
```

### Truncation Settings

Enable and configure truncation for long conversations:
```bash
enable_truncation=true
truncation_strategy="ast_llm_compaction"  # or "first_user_priority"
truncation_max_tokens=16000
```

## Dataset Format

The dataset should provide problems/tasks that require multi-step code execution:

```json
{
  "problem": "Write a function to calculate fibonacci numbers and test it",
  "instance_id": "fib_001",
  "ground_truth": "Expected output or answer",
  "test_cases": ["test cases if available"]
}
```

The `OrchestratorDataset` class will transform this into the appropriate format with:
- `agent_name: "orchestrator_coding_agent"` to select the correct agent
- Modal sandbox configuration (instance_id, run_id, notebook_id)
- Truncation settings

## Running Training

```bash
# Make the script executable
chmod +x run_orchestrator_qwen2_7b.sh

# Run training
./run_orchestrator_qwen2_7b.sh
```

## Customization

### Custom Dataset Processing

Modify `orchestrator.py` to handle your specific dataset format:

```python
def map_fn(self, row: dict):
    # Custom logic to extract problem and format prompt
    pass
```

### Custom Reward Function

The `compute_score` function in `orchestrator.py` can be customized for your evaluation needs:

```python
async def compute_score(data_dict: dict, response: Union[str, list[dict]], 
                       grader: Grader = None, **kwargs) -> float:
    # Custom scoring logic
    pass
```

### Agent Loop Modifications

The agent loop implementation is in `/verl/experimental/agent_loop/orchestrator_coding_agent_loop.py`. 
Key customization points:
- Code extraction pattern (currently `<code>...</code>`)
- Sandbox initialization parameters
- Response processing logic
- Custom truncation strategies

## Differences from Tool Agent (Retool)

| Feature | Tool Agent | Orchestrator Agent |
|---------|------------|--------------------|
| Code Execution | Stateless function calls | Stateful notebook |
| Message Format | Tool schemas + responses | Code blocks in messages |
| Environment | Fresh for each call | Persistent across conversation |
| Truncation | Not implemented | AST-based compaction |
| Use Case | Math problems, single computations | Complex coding tasks, debugging |

## Troubleshooting

1. **Import Errors**: Ensure httpx is installed: `pip install httpx`
2. **Modal API Issues**: Check Modal endpoint URL and authentication
3. **OOM Errors**: Reduce `max_response_length` or enable truncation
4. **Truncation Issues**: Verify tokenizer compatibility with truncation strategies
