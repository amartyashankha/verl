import re
import json
import ast
from typing import Dict, List, Optional, Set, Tuple
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import torch
from transformers import AutoTokenizer
from collections import defaultdict
from claude_thinking_python import ClaudeThinkingClient
import psutil
import time
import logging

logger = logging.getLogger(__name__)

def log_memory_usage(context: str = ""):
    """Log current memory usage for debugging"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        logger.info(f"MEMORY [{context}]: "
                   f"RSS={memory_info.rss / 1024 / 1024:.1f}MB, "
                   f"VMS={memory_info.vms / 1024 / 1024:.1f}MB, "
                   f"Process={memory_percent:.1f}%, "
                   f"System={system_memory.percent:.1f}% "
                   f"(Available={system_memory.available / 1024 / 1024 / 1024:.1f}GB)")
    except Exception as e:
        logger.warning(f"Failed to log memory usage [{context}]: {e}")

# Configuration
COMPACTION_THRESHOLD = 20000
# Create base compacted message list
main_agent_identity_text = "My identity is the main agent."
sub_agent_identity_text = "My identity is the subagent."




# Configuration
def truncate_strategy_1_first_user_priority(messages: List[Dict], tokenizer, max_tokens: int, is_subleader: bool, **kwargs) -> List[Dict]:
    """
    Strategy 1: First user query message priority 10, all other messages priority 9 - age
    Remove messages based on priority until total tokens < 16000
    Only processes context messages (not output messages)
    """
    if not messages:
        return messages
    
    # Process only context messages (all messages are context, no output message to exclude)
    context_messages = messages
    
    # Calculate priorities with indices for context messages only
    priorities = []
    for i, msg in enumerate(context_messages):
        
        if i == 0 and msg['role'] == 'user':
            priority = 10  # First user message gets highest priority
        else:
            # Age = how old the message is. Older messages (lower index) have higher age
            # Index 0 = oldest (age = len-1), Index len-1 = newest (age = 0)
            age = (len(context_messages) - 1) - i
            priority = 9 - age  # Newer messages (lower age) get higher priority
        priorities.append((priority, i, msg))
    
    # Sort by priority (descending)
    priorities.sort(key=lambda x: x[0], reverse=True)
    max_num_lastest_messages = 4
    # Keep adding messages until we exceed token limit
    selected_indices = set()
    current_tokens = 0
    
    for priority, idx, msg in priorities:
        # Add this message index to selected set
        test_indices = selected_indices | {idx}
        
        # Create test conversation with selected messages in original order
        test_messages = [context_messages[i] for i in sorted(test_indices)]
        
        # Tokenize to check length
        input_ids = tokenize_conversation(test_messages, tokenizer)
        test_tokens = len(input_ids)
        
        if test_tokens <= max_tokens:
            selected_indices.add(idx)
            current_tokens = test_tokens
            if len(selected_indices) >= max_num_lastest_messages:
                break
        else:
            break
    
    assert current_tokens <= max_tokens, f"ERROR: Strategy 1: Kept {len(selected_indices)}/{len(messages)} messages, ~{current_tokens} tokens"
    
    # Create result in original order
    result = [context_messages[i] for i in sorted(selected_indices)]
    
    # Fallback to ast_llm_compaction if result has less than 3 messages
    if len(result) < 3:
        print(f"Strategy 1 result has only {len(result)} messages, falling back to truncate_strategy_4_ast_llm_compaction")
        return truncate_strategy_4_ast_llm_compaction(messages, is_subleader, **kwargs)
    
    print(f"Strategy 1: Kept {len(result)}/{len(messages)} messages, ~{current_tokens} tokens")
    return result

def tokenize_conversation(conversation_messages, tokenizer):
    """Tokenize a conversation up to the current point"""
    
    # Normalize message content
    processed_conversation = []
    for message in conversation_messages:
        assert message['role'] in ['user', 'assistant'], f"Invalid role: {message['role']}"
        if isinstance(message['content'], list):
            content = message['content'][0]['text']
        else:
            content = message['content']
        assert isinstance(content, str), f"Invalid content: {content}"
        processed_conversation.append({"role": message['role'], "content": content})
    
    # Get tokens including current message
    tokens_with_current = tokenizer.apply_chat_template(
        processed_conversation,
        tokenize=True,
        add_generation_prompt=False
    )
    
    input_ids = tokens_with_current
    return input_ids

def truncate_conversation(messages: List[Dict], tokenizer, strategy: str = "first_user_priority", max_tokens: int = 16000, is_subleader: bool = False, **kwargs) -> List[Dict]:
    """
    Main truncation function that applies different strategies
    
    Strategies:
    - "first_user_priority": First user message priority 10, others 9-age
    - "ast_llm_compaction": Use AST analysis + LLM summarization with code-aware context
    """
    
    print(f"Truncating conversation: {len(messages)} messages -> strategy: {strategy}")
    
    if strategy == "first_user_priority":
        return truncate_strategy_1_first_user_priority(messages, tokenizer, max_tokens, is_subleader, **kwargs)
    elif strategy == "ast_llm_compaction":
        return truncate_strategy_4_ast_llm_compaction(messages, is_subleader, **kwargs)
    else:
        raise ValueError(f"Unknown strategy {strategy}")



def _extract_code_from_response(response: str) -> Optional[str]:
    """Extract code from <code></code> blocks in the response"""
    
    # Stack-based extraction for nested <code> tags
    extracted_blocks = []
    i = 0
    
    while i < len(response):
        # Look for the start of a <code> block
        if response[i:i+6] == '<code>':
            # Start extraction
            opening_positions = [i + 6]  # Stack of opening positions after <code>
            i += 6
            
            # Track any completed inner blocks
            inner_blocks = []
            
            # Find matching closing tag
            while i < len(response) and opening_positions:
                if response[i:i+6] == '<code>':
                    opening_positions.append(i + 6)
                    i += 6
                elif response[i:i+7] == '</code>':
                    if opening_positions:
                        start = opening_positions.pop()
                        content = response[start:i]
                        
                        if not opening_positions:
                            # This is the outermost block closing
                            extracted_blocks.append(content)
                        else:
                            # This is an inner block - save it
                            inner_blocks.append(content)
                    i += 7
                else:
                    i += 1
            
            # If we have unclosed tags but found inner blocks, use them
            if opening_positions and inner_blocks:
                extracted_blocks.extend(inner_blocks)
        else:
            i += 1
    
    if extracted_blocks:
        # Join all extracted code blocks with newlines
        code = '\n\n'.join(block.strip() for block in extracted_blocks if block.strip())
        return code
    
    # Alternative: look for ```python...``` blocks
    python_pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    if matches:
        code = '\n\n'.join(match.strip() for match in matches)
        return code
        
    # Alternative: look for ```...``` blocks
    code_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    if matches:
        code = '\n\n'.join(match.strip() for match in matches)
        return code
    
    return None

class GlobalStateAnalyzer(ast.NodeVisitor):
    """AST visitor to extract global state information from code"""
    
    def __init__(self):
        self.global_variables = set()
        self.functions = set()
        self.classes = set()
        self.imports = set()
        self.from_imports = set()
        self.import_aliases = {}
        
        # Track scope level (0 = global)
        self.scope_level = 0
        
    def visit_Import(self, node):
        """Track import statements"""
        if self.scope_level == 0:
            for alias in node.names:
                if alias.asname:
                    self.imports.add(f"import {alias.name} as {alias.asname}")
                    self.import_aliases[alias.asname] = alias.name
                else:
                    self.imports.add(f"import {alias.name}")
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track from ... import statements"""
        if self.scope_level == 0:
            module = node.module or ''
            for alias in node.names:
                if alias.name == '*':
                    self.from_imports.add(f"from {module} import *")
                elif alias.asname:
                    self.from_imports.add(f"from {module} import {alias.name} as {alias.asname}")
                    self.import_aliases[alias.asname] = f"{module}.{alias.name}"
                else:
                    self.from_imports.add(f"from {module} import {alias.name}")
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Track function definitions"""
        if self.scope_level == 0:
            # Extract parameter names
            params = []
            for arg in node.args.args:
                params.append(arg.arg)
            param_str = ", ".join(params) if params else ""
            self.functions.add(f"{node.name}({param_str})")
        
        # Enter function scope
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1
        
    def visit_AsyncFunctionDef(self, node):
        """Track async function definitions"""
        if self.scope_level == 0:
            # Extract parameter names
            params = []
            for arg in node.args.args:
                params.append(arg.arg)
            param_str = ", ".join(params) if params else ""
            self.functions.add(f"async {node.name}({param_str})")
        
        # Enter function scope
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1
        
    def visit_ClassDef(self, node):
        """Track class definitions"""
        if self.scope_level == 0:
            # Extract base classes if any
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    # Handle things like nn.Module
                    parts = []
                    current = base
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    bases.append(".".join(reversed(parts)))
            
            base_str = f"({', '.join(bases)})" if bases else ""
            self.classes.add(f"{node.name}{base_str}")
        
        # Enter class scope
        self.scope_level += 1
        self.generic_visit(node)
        self.scope_level -= 1
        
    def visit_Assign(self, node):
        """Track global variable assignments"""
        if self.scope_level == 0:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.global_variables.add(target.id)
                elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                    # Handle unpacking like a, b = ...
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.global_variables.add(elt.id)
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node):
        """Track annotated assignments like x: int = 5"""
        if self.scope_level == 0:
            if isinstance(node.target, ast.Name):
                self.global_variables.add(node.target.id)
        self.generic_visit(node)
        
    def visit_AugAssign(self, node):
        """Track augmented assignments like x += 1"""
        if self.scope_level == 0:
            if isinstance(node.target, ast.Name):
                self.global_variables.add(node.target.id)
        self.generic_visit(node)

def analyze_code_for_globals(code: str) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Analyze code to extract global state information
    Returns: (variables, functions, classes, imports)
    """
    try:
        tree = ast.parse(code)
        analyzer = GlobalStateAnalyzer()
        analyzer.visit(tree)
        
        # Combine all imports
        all_imports = analyzer.imports | analyzer.from_imports
        
        return (
            analyzer.global_variables,
            analyzer.functions,
            analyzer.classes,
            all_imports
        )
    except SyntaxError:
        # If code has syntax errors, return empty sets
        return set(), set(), set(), set()

def generate_summary_prompt_with_ast_context(variables: Set[str], functions: Set[str], classes: Set[str], imports: Set[str], message_count: int) -> str:
    """Generate the summarization prompt with AST analysis as context"""
    
    # Format the AST analysis results
    ast_context_parts = []
    
    if variables:
        variables_str = ", ".join(sorted(variables))
        ast_context_parts.append(f"Variables defined: {variables_str}")
    
    if functions:
        functions_str = ", ".join(sorted(functions))
        ast_context_parts.append(f"Functions defined: {functions_str}")
    
    if classes:
        classes_str = ", ".join(sorted(classes))
        ast_context_parts.append(f"Classes defined: {classes_str}")
    
    if imports:
        imports_str = ", ".join(sorted(imports))
        ast_context_parts.append(f"Imports: {imports_str}")
    
    ast_context = "\n".join(ast_context_parts) if ast_context_parts else "No code elements found"
    
    return f"""Your task is to create a comprehensive document of the conversation so far. This conversation involves generating Python code in an IPython notebook to solve a problem.

Each assistant message containing a `<code>` block is added as a new cell to the notebook. The user's messages (after the first one) represent the `stdout` and `stderr` from executing that code.
CRITICAL: Only `<code>` tags are code blocks. Do NOT treat markdown code blocks like ```Python or ```python as actual code blocks. These are just formatting and should be ignored.

AST ANALYSIS CONTEXT:
The following code elements have been identified through static analysis, these are the only code element you should consider:
{ast_context}

Total messages processed: {message_count}

IMPORTANT: Use the AST analysis context above to help you understand what code elements exist. When documenting variables, functions, classes, and imports, provide detailed descriptions of their purpose and what they contain based on the conversation context.

Your document should be comprehensive and include:

1. **Problem Definition**: What problem is being solved? Be detailed about the original requirements and goals.

2. **Past Work**: What has been done so far? Include:
   - ALL operations performed (data loading, processing, model calls, etc.) with specific details
   - Any LLM calls made and their purposes, including prompts used and responses received
   
   **Code Documentation**:
   - For each variable: `variable_name`: Data type describing what it contains and its purpose.
   - For each import: `import_name`: Description of what the imported module/function/class provides.
   - For each function: `function_name(parameters)`: Return type describing what the function does.
   - For each class: `class_name`: Description of the class purpose and its methods.

<analysis>
[Your chronological analysis here - go through each message one by one with a good summary of what was done, group them when appropriate, don't make it too long]
</analysis>

<summary>
1. Problem Definition:
[Comprehensive summary of the problem statement]

2. Past Work:
[Comprehensive summary of all operations performed]

**Code Documentation**:
[Format all variables, imports, functions, and classes as: `name`: Data type describing what it contains and its purpose.]

</summary>
"""

def format_summary(summary_text: str) -> str:
    """Format the summary by extracting analysis and summary tags"""
    result = summary_text
    
    # Extract and format analysis section
    analysis_match = re.search(r'<analysis>([\s\S]*?)</analysis>', result, re.IGNORECASE)
    if analysis_match:
        analysis_content = analysis_match.group(1) or ""
        result = re.sub(r'<analysis>[\s\S]*?</analysis>', 
                       f'Analysis:\n{analysis_content.strip()}', result, flags=re.IGNORECASE)
    
    # Extract and format summary section
    summary_match = re.search(r'<summary>([\s\S]*?)</summary>', result, re.IGNORECASE)
    if summary_match:
        summary_content = summary_match.group(1) or ""
        # Escape single backslashes to prevent re.error
        summary_content = summary_content.replace('\\', '\\\\')
        result = re.sub(r'<summary>[\s\S]*?</summary>', 
                       f'Summary:\n{summary_content.strip()}', result, flags=re.IGNORECASE)
    
    return result.strip()

def truncate_strategy_4_ast_llm_compaction(messages: List[Dict], is_subleader: bool, **kwargs) -> List[Dict]:
    """Compact the conversation using AST analysis + LLM summarization"""
    
    # Track cumulative global state through AST analysis
    all_variables = set()
    all_functions = set()
    all_classes = set()
    all_imports = set()
    
    user_count = 0
    assistant_count = 0
    
    # Go through every message for AST analysis
    for i, message in enumerate(messages):
        role = message['role']
        
        # Count messages
        if role == 'user':
            user_count += 1
        elif role == 'assistant':
            assistant_count += 1
        
        # Extract content
        if isinstance(message['content'], list):
            content = message['content'][0]['text'] if message['content'] else ""
        else:
            content = message['content']
        
        # Only process assistant messages for code extraction
        if (role == 'assistant' and content) or (role == 'user' and i == 0):
            # Extract code from the message
            code = _extract_code_from_response(content)
            
            if code:
                # Analyze the code for global state
                variables, functions, classes, imports = analyze_code_for_globals(code)
                
                # Update cumulative state
                all_variables.update(variables)
                all_functions.update(functions)
                all_classes.update(classes)
                all_imports.update(imports)
    
    # Generate summary prompt with AST context
    summary_prompt = generate_summary_prompt_with_ast_context(
        all_variables, all_functions, all_classes, all_imports, len(messages)
    )
    
    # Use LLM to generate summary
    claude_client = ClaudeThinkingClient()
    summary_messages = messages + [{
        "role": "user",
        "content": summary_prompt
    }]
    
    summary_response = claude_client.call_llm(
        messages=summary_messages,
        system="You are a helpful assistant that summarizes conversations.",
        model="claude-sonnet-4-20250514",
        temperature=1.0,
        max_tokens=20000,
        tools=None,
        thinking_budget_tokens=1024,
        stream=False
    )
    summary_text = summary_response.get("output", "")

    # Format the summary
    formatted_summary = format_summary(summary_text)
    
    # Calculate original message size
    original_size = len(json.dumps(messages))
    
  
    identity_str = sub_agent_identity_text if is_subleader else main_agent_identity_text
    base_compacted_messages = [{
        "role": "user",
        "content": identity_str + f"\n\nThis session is being continued from a previous conversation. Summary:\n\n{formatted_summary}"
    }]
    base_compacted_size = len(json.dumps(base_compacted_messages))
    
    # Try different numbers of recent messages (3, 2, 1, 0) - descending order
    max_possible_recent = min(3, len(messages))
    optimal_recent_count = 0
    
    print(f"Original messages size: {original_size} chars")
    print(f"Base compacted size: {base_compacted_size} chars")
    
    for num_recent in range(max_possible_recent, -1, -1):  # Descending from max to 0
        if num_recent == 0:
            test_compacted = base_compacted_messages.copy()
            recent_messages = []
        else:
            recent_messages = messages[-num_recent:]
            test_compacted = base_compacted_messages + recent_messages
        
        test_size = len(json.dumps(test_compacted))
        reduction_percentage = (original_size - test_size) / original_size * 100
        
        print(f"With {num_recent} recent messages: {test_size} chars, {reduction_percentage:.1f}% reduction")
        
        # Check if we achieve at least 50% reduction
        if reduction_percentage >= 50.0:
            optimal_recent_count = num_recent
            print(f"Taking {num_recent} recent messages (first to achieve >=50% reduction)")
            break
    
    # Use the optimal number of recent messages
    if optimal_recent_count == 0:
        compacted_messages = base_compacted_messages.copy()
        recent_messages = []
    else:
        recent_messages = messages[-optimal_recent_count:]
        compacted_messages = base_compacted_messages + recent_messages
    
    print(f"Selected {optimal_recent_count} recent messages for optimal compaction")
    
    # Print summary
    print(f"AST+LLM compaction complete:")
    print(f"  Original messages: {len(messages)}")
    print(f"  User messages: {user_count}")
    print(f"  Assistant messages: {assistant_count}")
    print(f"  Variables found: {len(all_variables)}")
    print(f"  Functions found: {len(all_functions)}")
    print(f"  Classes found: {len(all_classes)}")
    print(f"  Imports found: {len(all_imports)}")
    print(f"  Compacted to: {len(compacted_messages)} messages")

    final_size = len(json.dumps(compacted_messages))
    final_reduction = (original_size - final_size) / original_size * 100
    print(f"Original size: {original_size} chars")
    print(f"Base compacted size: {base_compacted_size} chars")
    print(f"Final compacted messages: {final_size} chars ({final_reduction:.1f}% reduction)")
    
    return compacted_messages

def tokenize_conversation_part(conversation_messages, tokenizer):
    """Tokenize a conversation up to the current point"""
    # Normalize message content
    processed_conversation = []
    for message in conversation_messages:
        assert message['role'] in ['user', 'assistant'], f"Invalid role: {message['role']}"
        if isinstance(message['content'], list):
            content = message['content'][0]['text']
        else:
            content = message['content']
        assert isinstance(content, str), f"Invalid content: {content}"
        processed_conversation.append({"role": message['role'], "content": content})
    
    # Get tokens for context (all but last message)
    tokens_context = tokenizer.apply_chat_template(
        processed_conversation[:-1],
        tokenize=True,
        add_generation_prompt=False
    )
    
    # Get tokens including current message
    tokens_with_current = tokenizer.apply_chat_template(
        processed_conversation,
        tokenize=True,
        add_generation_prompt=False
    )
    
    num_prompt_tokens = len(tokens_context)
    assistant_message_tokens_count = len(tokens_with_current) - num_prompt_tokens
    
    # Create loss mask
    current_loss_mask = [0] * num_prompt_tokens + [1] * assistant_message_tokens_count
    
    input_ids = tokens_with_current[:-1]
    labels = tokens_with_current[1:]
    loss_mask = current_loss_mask[1:]
    
    assert len(input_ids) == len(labels) == len(loss_mask)
    
    return input_ids, labels, loss_mask

def process_conversation(enumerated_conversation, tokenizer, strategy):
    """Processes a single conversation."""
    conv_id, conversation = enumerated_conversation



    
    first_text = conversation[0]['content'][0]["text"]

    is_subleader = sub_agent_identity_text in first_text

    if sub_agent_identity_text in first_text:
        is_subleader = True
    elif main_agent_identity_text in first_text:
        is_subleader = False
    else:
        raise ValueError(f"Invalid first text: {first_text}")

    print(f"Processing conversation {conv_id}")
    
    processed_data_for_conv = []
    
    # Initialize compression state
    cur_compressed_context = []
    cur_compressed_assistant_message_index = 0
    
    # Find all assistant message indices
    assistant_indices = [i for i, msg in enumerate(conversation) if msg['role'] == 'assistant']
    
    # Process each assistant message
    for msg_idx in assistant_indices:
        # Build real context
        if cur_compressed_context:
            # Use compressed context + recent messages
            real_context = (cur_compressed_context + 
                        conversation[cur_compressed_assistant_message_index:msg_idx])
        else:
            # Use original context
            real_context = conversation[:msg_idx]
        
        # Add current assistant message
        real_all = real_context + [conversation[msg_idx]]
        
        # Tokenize with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                compacted = False
                input_ids, labels, loss_mask = tokenize_conversation_part(real_all, tokenizer)
                print(f"  Tokenized for conv {conv_id} at message {msg_idx}, length: {len(input_ids)}")
                # Check if compaction needed
                if len(input_ids) > COMPACTION_THRESHOLD:
                    print(f"  Compacting for conv {conv_id} at message {msg_idx}, length: {len(input_ids)}")
                    compacted = True
                    
                    # Compact the current real_context using AST+LLM analysis
                    cur_compressed_context = truncate_conversation(real_context, tokenizer, strategy, max_tokens=COMPACTION_THRESHOLD-2000, is_subleader=is_subleader)
                    cur_compressed_assistant_message_index = msg_idx
                    
                    print(f"  Compressed for conv {conv_id} from {len(json.dumps(real_context))} to {len(json.dumps(cur_compressed_context))} chars")
                    real_context = cur_compressed_context
                    real_all = real_context + [conversation[msg_idx]]
                    input_ids, labels, loss_mask = tokenize_conversation_part(real_all, tokenizer)

                # Store processed data
                processed_item = {
                    "conversation_id": conv_id,
                    "assistant_message_index": msg_idx,
                    "input_ids": input_ids,
                    "labels": labels,
                    "loss_mask": loss_mask,
                    "real_context": real_context,
                    "assistant_message": conversation[msg_idx],
                    "compacted": compacted
                }
                processed_data_for_conv.append(processed_item)
                
                # If successful, break the retry loop
                break
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  Error processing conversation {conv_id}, message {msg_idx} on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt + 1 < max_retries:
                    print("  Retrying...")
        else:
            print(f"  Failed after {max_retries} attempts for conv {conv_id}. Skipping message {msg_idx}.")
            continue
    return processed_data_for_conv

def main():
    parser = argparse.ArgumentParser(description="Process conversation data using AST analysis + LLM summarization for compaction.")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of worker processes to use.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of conversations to process.")
    parser.add_argument("--conversation-path", type=str, default=None, help="Path to a specific conversation JSON file to process instead of default train/test files.")
    parser.add_argument("--strategy", type=str, default="first_user_priority", help="Strategy to use for compaction.")
    args = parser.parse_args()

    # Load tokenizer
    log_memory_usage("Before tokenizer loading in compact_truncate")
    tokenizer_start_time = time.time()
    logger.info(f"TOKENIZER: Loading Qwen/Qwen2.5-Coder-7B-Instruct in compact_truncate...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    tokenizer_duration = time.time() - tokenizer_start_time
    log_memory_usage("After tokenizer loading in compact_truncate")
    logger.info(f"TOKENIZER: Loaded in compact_truncate in {tokenizer_duration:.1f}s")
    
    # Check if a specific conversation path is provided
    if args.conversation_path:
        # Load specific conversation file
        print(f"Loading specific conversation from: {args.conversation_path}")
        with open(args.conversation_path, "r") as f:
            conversation = json.load(f)
        
        # Set as single conversation list
        original_conversations = [conversation["messages"]]
        
        # Create output directory and filename based on the conversation path
        conversation_dir = os.path.dirname(os.path.dirname(args.conversation_path))
        conversation_filename = os.path.basename(args.conversation_path)
        output_dir = f"{conversation_dir}/{args.strategy}_output_max_tokens_{COMPACTION_THRESHOLD}_single"
        os.makedirs(output_dir, exist_ok=True)
        
        # Change extension to .jsonl
        output_filename = conversation_filename.replace('.json', '.jsonl')
        output_path = f"{output_dir}/{output_filename}"
        
        print(f"Loaded 1 conversation from {args.conversation_path}")
        
        # Create a partial function with the tokenizer
        process_func = partial(process_conversation, tokenizer=tokenizer, strategy=args.strategy)
        
        with Pool(processes=args.max_workers) as pool:
            # Process conversations in parallel
            results = pool.map(process_func, enumerate(original_conversations))

        # Save processed data - each conversation as one line in JSONL
        total_items = 0
        filtered_items = 0
        with open(output_path, "w") as f:
            for conv_data in results:
                if conv_data:  # Only write non-empty conversation data
                    # Write each data item as a separate JSON line
                    for item in conv_data:
                        # Filter out items with input_ids > COMPACTION_THRESHOLD
                        if len(item['input_ids']) <= COMPACTION_THRESHOLD:
                            json.dump(item, f)
                            f.write('\n')
                            total_items += 1
                        else:
                            filtered_items += 1
                            print(f"Filtered out item with {len(item['input_ids'])} tokens (threshold: {COMPACTION_THRESHOLD})")
        print(f"\nSaved {len(results)} conversations with {total_items} total items to {output_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"  Total conversations: {len(original_conversations)}")
        print(f"  Total processed items: {total_items}")
        print(f"  Filtered out items: {filtered_items}")
        
        # Count compactions across all conversations
        compaction_count = 0
        for conv_data in results:
            if conv_data:
                compaction_count += sum(1 for item in conv_data if item.get('compacted'))
        print(f"  Items requiring compaction: {compaction_count}")
        
    else:
        # Load data from default files
        base_path = "/home/tianhangzhu/gcs_view/home/tianhangzhu/data/noninteractive_results_swe_gym_train_rest_v2/"
        files = ["leader_train.json", "leader_test.json"]
        
        for file in files:
            # Load original conversations
            with open(f"{base_path}/{file}", "r") as f:
                original_conversations = json.load(f)
            
            if args.limit:
                if "train" in file:
                    original_conversations = original_conversations[:args.limit]
                else:
                    limit = int(args.limit * 0.1)
                    original_conversations = original_conversations[:limit]

            print(f"Loaded {len(original_conversations)} conversations from {file}")
            
            # Create a partial function with the tokenizer
            process_func = partial(process_conversation, tokenizer=tokenizer, strategy=args.strategy)
            
            with Pool(processes=args.max_workers) as pool:
                # Process conversations in parallel
                results = pool.map(process_func, enumerate(original_conversations))

            # Save processed data - each conversation as one line in JSONL
            output_dir = f"{base_path}/{args.strategy}_output_max_tokens_{COMPACTION_THRESHOLD}_{args.limit}convs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Change extension to .jsonl
            output_filename = file.replace('.json', '.jsonl')
            output_path = f"{output_dir}/{output_filename}"
            
            # Write each conversation's data as a separate line
            total_items = 0
            filtered_items = 0
            with open(output_path, "w") as f:
                for conv_data in results:
                    if conv_data:  # Only write non-empty conversation data
                        # Write each data item as a separate JSON line
                        for item in conv_data:
                            # Filter out items with input_ids > COMPACTION_THRESHOLD
                            if len(item['input_ids']) <= COMPACTION_THRESHOLD:
                                json.dump(item, f)
                                f.write('\n')
                                total_items += 1
                            else:
                                filtered_items += 1
                                print(f"Filtered out item with {len(item['input_ids'])} tokens (threshold: {COMPACTION_THRESHOLD})")
            print(f"\nSaved {len(results)} conversations with {total_items} total items to {output_path}")
            
            # Print summary statistics
            print(f"\nSummary for {file}:")
            print(f"  Total conversations: {len(original_conversations)}")
            print(f"  Total processed items: {total_items}")
            print(f"  Filtered out items: {filtered_items}")
            
            # Count compactions across all conversations
            compaction_count = 0
            for conv_data in results:
                if conv_data:
                    compaction_count += sum(1 for item in conv_data if item.get('compacted'))
            print(f"  Items requiring compaction: {compaction_count}")

if __name__ == "__main__":
    main()