# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
import re  # Added for code block extraction
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

# TOREVIEW (Shankha): Import httpx for Modal API calls
import httpx
from datasets import load_dataset

# TOREVIEW (Shankha): Import truncation utilities for AST-based compaction
from verl.experimental.agent_loop.utils.compact_truncate import truncate_conversation
    
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
# from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser  # TOREVIEW (Shankha): Commented out - not using tool parsing
from verl.tools.schemas import ToolResponse
# from verl.tools.utils.tool_registry import initialize_tools_from_config  # TOREVIEW (Shankha): Commented out - not using tool registry
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Ensure logger outputs to console
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# Global variable to store the log file path for this run
_current_log_file = None

def save_llm_interaction(messages, output_text, log_file_path=None):
    """Save LLM input messages and output to JSON file"""
    global _current_log_file
    
    if log_file_path is None:
        if _current_log_file is None:
            import time
            import os
            # Create timestamped filename in ~/RLlog/ - once per run
            timestamp = int(time.time())
            log_dir = os.path.expanduser("~/RLlog")
            os.makedirs(log_dir, exist_ok=True)
            _current_log_file = os.path.join(log_dir, f"test_llm_{timestamp}.json")
        log_file_path = _current_log_file
    
    try:
        import time
        interaction_data = {
            "timestamp": time.time(),
            "messages": messages,
            "output": output_text
        }
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Read existing data if file exists
        existing_data = []
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        existing_data = json.loads(content)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing log file: {e}")
                existing_data = []
        
        # Append new data
        existing_data.append(interaction_data)
        
        # Write back to file
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved LLM interaction to {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to save LLM interaction: {e}")


def build_full_prompt_for_first_cell(task_prompt: str) -> tuple[str, str]:
    """Build the full prompt exactly like run_leader_agent_swebench.py.
    Returns (full_prompt_for_starting_txt, first_cell_source_code).
    """
    # 1) Load tool_definitions.json from the same location as SWE-bench runner
    def _load_tool_definitions() -> list[dict]:
        # Primary location: same directory as this file
        tool_definitions_path = Path(__file__).parent / "tool_definitions.json"
        #/home/tianhangzhu/RL/jeffery/verl/verl/experimental/agent_loop/orchestrator_coding_agent_loop.py
        #/home/tianhangzhu/RL/jeffery/hierarchial_ppo_shankha/inference/tool_definitions.json
        # Fallback locations if not found
        candidates = [
            tool_definitions_path,
            Path(__file__).parent.parent.parent.parent.parent / "hierarchial_ppo_shankha" / "inference" / "tool_definitions.json",
        ]
        
        for p in candidates:
            try:
                if p.exists():
                    with open(p, 'r') as f:
                        return json.load(f)
            except Exception:
                pass
        
        # Return empty list if not found (matching run_leader_agent_swebench.py behavior)
        return []

    def _convert_js_to_python(obj):
        """Recursively convert JavaScript-style values to Python equivalents"""
        if isinstance(obj, dict):
            return {key: _convert_js_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_convert_js_to_python(item) for item in obj]
        else:
            # Handle boolean values - check both actual booleans and string representations
            if obj is True:
                return True
            elif obj is False:
                return False
            elif obj is None:
                return None
            elif isinstance(obj, str):
                obj_lower = obj.lower().strip()
                if obj_lower == "false":
                    return False
                elif obj_lower == "true":
                    return True
                elif obj_lower == "null":
                    return None
            return obj

    def _json_dumps_python_style(obj):
        """JSON dumps that preserves Python boolean format"""
        json_str = json.dumps(obj, indent=2)
        # Replace JSON booleans with Python booleans
        json_str = json_str.replace('"true"', 'True')
        json_str = json_str.replace('"false"', 'False')
        json_str = json_str.replace('"null"', 'None')
        json_str = json_str.replace('true', 'True')
        json_str = json_str.replace('false', 'False')
        json_str = json_str.replace('null', 'None')
        return json_str

    # Convert tool definitions
    tool_defs = _convert_js_to_python(_load_tool_definitions())
    tool_def_str = _json_dumps_python_style(tool_defs)

    # Double-JSON-dump for problem_statement (exactly as in run_leader_agent_swebench.py)
    escaped_problem_statement = json.dumps(json.dumps(task_prompt))

    # Create first cell source using triple-quoted f-string format (identical to run_leader_agent_swebench.py)
    first_cell_source = f'''# Problem Statement and Configuration
problem_statement = {escaped_problem_statement}

work_dir = "/testbed"

# Tool Definitions
tool_definitions = {tool_def_str}'''

    # Wrap the code in <code> tags
    wrapped_code = f"<code>{first_cell_source}</code>"
    
    # Return the prompt in the expected format
    full_prompt = f"<thinking>Please solve a PR request</thinking>{wrapped_code}"
    
    return full_prompt, first_cell_source


@register("orchestrator_coding_agent")
class OrchestratorCodingAgentLoop(AgentLoopBase):
    """
    TOREVIEW (Shankha): Modal notebook agent with AST-based truncation support
    
    Key design decisions:
    1. Tracks conversation as messages throughout the loop (not just tokens)
    2. Applies truncation before generation when context gets too long
    3. Regenerates response_mask and log_probs after truncation (simpler than tracking)
    4. Uses the same truncation utilities as inference (leader_agent.py)
    
    Flow:
    1. Dataset provides agent_name="orchestrator_coding_agent" in non_tensor_batch
    2. AgentLoopWorker looks up this agent in _agent_loop_registry 
    3. This class is instantiated with VERL's tokenizer/processor
    4. Messages are tokenized using VERL's approach (supports multi-modal)
    5. Truncation strategies use their own tokenization (acceptable mismatch)
    
    TODO (Shankha): Consider these improvements:
    1. Cache tokenization results to avoid re-tokenizing after truncation
    2. Track response_mask segments to preserve them through truncation
    3. Add configuration for when to trigger truncation (e.g., buffer before max)
    4. Handle multi-modal data through truncation (currently might lose images)
    """
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level OrchestratorCodingAgentLoop initialization")
        
        # Reset the global log file for this new run
        global _current_log_file
        _current_log_file = None

        # TOREVIEW (Shankha): Initialize basic attributes
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        # cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls  # TOREVIEW (Shankha): Not needed for sequential notebook execution
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length  # TOREVIEW (Shankha): Reuse for output length limit
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side  # TOREVIEW (Shankha): Reuse for output truncation
        
        # TOREVIEW (Shankha): Tool initialization commented out - using Modal API instead
        # tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        # tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        # cls.tools = {tool.name: tool for tool in tool_list}
        # cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        # cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        # print(f"Initialized tools: {cls.tools}")
        
        # TOREVIEW (Shankha): Initialize Modal-specific configuration
        cls.modal_base_url = config.actor_rollout_ref.rollout.multi_turn.get("modal_base_url", "https://fairies--incremental-leader-agent-api")
        cls.modal_timeout = config.actor_rollout_ref.rollout.multi_turn.get("modal_timeout", 300)
        cls.code_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)  # TOREVIEW (Shankha): Pattern to extract code blocks
        
        # NEW: Modal endpoint configuration for text generation
        cls.use_modal_endpoint = config.actor_rollout_ref.rollout.multi_turn.get("use_modal_endpoint", False)
        cls.modal_chat_endpoint = config.actor_rollout_ref.rollout.multi_turn.get(
            "modal_chat_endpoint", 
            "https://fairies--vllm-global-step-900-cons-7514578f-serve.modal.run/v1/chat/completions"
        )
        cls.modal_model_name = config.actor_rollout_ref.rollout.multi_turn.get(
            "modal_model_name", 
            "vllm-global-step-900-cons-7514578f"
        )
        
        print(f"Initialized Modal agent with base URL: {cls.modal_base_url}")
        if cls.use_modal_endpoint:
            print(f"Using Modal endpoint for text generation: {cls.modal_chat_endpoint}")
        else:
            print("Using server_manager for text generation")

        # Normalize modal endpoint helpers to avoid malformed hostnames
        def _compose_endpoint(path_suffix: str) -> str:
            base = cls.modal_base_url.rstrip('/')
            # Modal converts function names with underscores to hyphens in URLs
            # e.g., init_sandbox -> init-sandbox
            # Remove any leading hyphens or slashes from the suffix
            normalized_suffix = path_suffix.lstrip('/-')
            if base.endswith('.modal.run'):
                # Full domain provided: use path-based endpoints
                return f"{base}/{normalized_suffix}"
            # Prefix provided: construct subdomain per Modal convention
            # Modal app name: incremental-leader-agent-api
            # Function: init_sandbox -> URL: incremental-leader-agent-api-init-sandbox.modal.run
            return f"{base}-{normalized_suffix}.modal.run"

        cls._endpoint = staticmethod(_compose_endpoint)
        
        # TOREVIEW (Shankha): Initialize truncation configuration for AST-based compaction
        cls.truncation_strategy = config.actor_rollout_ref.rollout.multi_turn.get("truncation_strategy", None)
        cls.truncation_max_tokens = config.actor_rollout_ref.rollout.multi_turn.get("truncation_max_tokens", 16000)
        cls.enable_truncation = config.actor_rollout_ref.rollout.multi_turn.get("enable_truncation", False)
        if cls.enable_truncation:
            print(f"Truncation enabled with strategy: {cls.truncation_strategy}, max_tokens: {cls.truncation_max_tokens}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        # TOREVIEW (Jeffrey) Initialize LRU cache for tokenization
        cls.enable_tokenization_cache = config.actor_rollout_ref.rollout.multi_turn.get("enable_tokenization_cache", True)
        cls._cached_apply_chat_template = lru_cache(maxsize=1000)( # could probably make maxsize larger if you wanted
            cls._apply_chat_template_worker
        )
    
    @classmethod
    def _apply_chat_template_worker(cls, messages_tuple, processor_kwargs_tuple):
        # Convert back from tuples (since lru_cache needs hashable arguments)
        messages = list(messages_tuple)
        processor_kwargs = dict(processor_kwargs_tuple)
        
        return cls.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **processor_kwargs,
        )

    async def _generate_with_modal_endpoint(self, messages, sampling_params, request_id):
        """Generate text using Modal chat completion endpoint"""
        from verl.workers.rollout.async_server import TokenOutput
        
        async with httpx.AsyncClient(timeout=self.modal_timeout) as client:
            # Prepare OpenAI-compatible request
            request_data = {
                "model": self.modal_model_name,
                "messages": messages,
                "max_tokens": sampling_params.get("max_tokens", 100),
                "temperature": sampling_params.get("temperature", 0.7),
                "top_p": sampling_params.get("top_p", 1.0),
                "stream": False
            }
            
            # Add logprobs if requested
            if sampling_params.get("logprobs"):
                request_data["logprobs"] = True
                request_data["top_logprobs"] = 5
            
            try:
                response = await client.post(
                    self.modal_chat_endpoint,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                response_data = response.json()
                
                # Extract response text
                choice = response_data["choices"][0]
                response_text = choice["message"]["content"]
                
                # Save LLM interaction to JSONL file
                save_llm_interaction(messages, response_text)
                
                # Convert response text to token IDs
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                
                # Extract logprobs if available
                log_probs = None
                if sampling_params.get("logprobs") and "logprobs" in choice:
                    log_probs = []
                    if choice["logprobs"] and "content" in choice["logprobs"]:
                        for token_info in choice["logprobs"]["content"]:
                            log_probs.append(token_info.get("logprob", 0.0))
                    
                    # Ensure log_probs matches response_ids length
                    while len(log_probs) < len(response_ids):
                        log_probs.append(0.0)
                    log_probs = log_probs[:len(response_ids)]
                
                return TokenOutput(
                    token_ids=response_ids,
                    log_probs=log_probs
                )
                
            except Exception as e:
                logger.error(f"Error calling Modal endpoint: {e}")
                # Fallback to empty response
                return TokenOutput(token_ids=[], log_probs=None)

    def extract_code_from_response(self, response: str) -> list[str]:
        """Extract code from <code></code> blocks in the response, return list of code blocks"""
        extracted_blocks = []
        i = 0
        
        while i < len(response):
            if response[i:i+6] == '<code>':
                opening_positions = [i + 6]
                i += 6
                inner_blocks = []
                
                while i < len(response) and opening_positions:
                    if response[i:i+6] == '<code>':
                        opening_positions.append(i + 6)
                        i += 6
                    elif response[i:i+7] == '</code>':
                        if opening_positions:
                            start = opening_positions.pop()
                            content = response[start:i]
                            
                            if not opening_positions:
                                extracted_blocks.append(content.strip())
                            else:
                                inner_blocks.append(content.strip())
                        i += 7
                    else:
                        i += 1
                
                if opening_positions and inner_blocks:
                    extracted_blocks.extend(inner_blocks)
            else:
                i += 1
        
        # Filter out empty blocks and return list
        return [block for block in extracted_blocks if block.strip()]
    def _extract_model_output_from_execution(self, exec_data: dict) -> dict:
        """Extract model output from execution result stdout that contains 'LLM Response:\\n:{model_output}' format"""
        try:
            stdout = exec_data.get('stdout', '') 
            stderr = exec_data.get('stderr', '')
            if not stdout or stderr:
                return None
                
            # Look for "LLM Response:" prefix and extract everything after it
            llm_prefix = "LLM Response:"
            if llm_prefix in stdout:
                # Extract everything after "LLM Response:"
                after_prefix = stdout.split(llm_prefix, 1)[1].strip()
                
                # Remove the leading ":" if present
                if after_prefix.startswith(':'):
                    after_prefix = after_prefix[1:].strip()
                
                # Stop at "Tool result message:" if it exists
                tool_result_prefix = "Tool result message:"
                if tool_result_prefix in after_prefix:
                    after_prefix = after_prefix.split(tool_result_prefix, 1)[0].strip()
                
                # Try to parse as JSON
                try:
                    import json
                    parsed = json.loads(after_prefix)
                    logger.info(f"Found model output in execution result: {parsed}")
                    return parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse execution result model output as JSON: {e}")
                    logger.warning(f"Raw content after LLM Response: {repr(after_prefix)}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting model output from execution: {e}")
            return None

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Sandbox handles full prompt generation; default to empty if not provided
        full_prompt, first_cell_code = build_full_prompt_for_first_cell(kwargs.get("task_prompt", ""))

        messages = [{"role": "user", "content": full_prompt}]
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        
        # TOREVIEW (Shankha): Track conversation messages for truncation
        # This mirrors the pattern from leader_agent.py
        conversation_messages = copy.deepcopy(messages)  # Keep track of full conversation
        
        # TOREVIEW (Shankha): Extract instance_id and run_id from kwargs
        instance_id = kwargs.get("instance_id", "default_instance")
        run_id = kwargs.get("run_id", f"run_{request_id}")
        notebook_id = kwargs.get("notebook_id", "main")
        task_prompt = kwargs.get("task_prompt", None)
        dataset_name = kwargs.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
        split = kwargs.get("split", ("train" if "swe-gym" in dataset_name.lower() else "test"))
        
        # TOREVIEW (Jeffrey): task_prompt now comes from the dataset
        # No need to load the dataset again here
        if not task_prompt:
            logger.warning(f"No task_prompt provided for {instance_id}. Modal sandbox will start without initial problem statement.")
        
        # TOREVIEW (Shankha): Check if httpx is available
        if httpx is None:
            raise ImportError("httpx is required for Modal notebook agent. Install it with: pip install httpx")
        
        # TOREVIEW (Shankha): Initialize HTTP client and sandbox
        async with httpx.AsyncClient(timeout=self.modal_timeout) as client:
            # TOREVIEW (Shankha): Initialize the sandbox for this agent
            init_response = await client.post(
                self._endpoint("init-sandbox"),  # Modal function: init_sandbox
                json={
                    "dataset": dataset_name,
                    "instance_id": instance_id,
                    "run_id": run_id,
                    "notebook_id": notebook_id,
                    "model_endpoint": "verl",  # TODO: Get from config if needed
                    "truncation_strategy": self.truncation_strategy or "ast_llm_compaction",
                    "max_tokens": self.truncation_max_tokens,
                    # Pass task_prompt to trigger first-cell creation and prompt file write
                    **{"full_prompt": full_prompt, "first_cell_code": first_cell_code}
                }
            )
            init_data = init_response.json()
            if not init_data.get("success"):
                logger.error(f"Failed to initialize Modal sandbox: {init_data}")
                # TODO: Decide how to handle initialization failure
                raise RuntimeError(f"Failed to initialize Modal sandbox: {init_data}")
            
            sandbox_id = init_data.get("sandbox_id")
            logger.info(f"Initialized Modal sandbox: {sandbox_id}")
        
        # TOREVIEW (Shankha): Apply truncation to initial messages if needed
        # This handles cases where the initial prompt itself is too long
        if self.enable_truncation:
            initial_tokens = await self._tokenize_messages(messages)
            if len(initial_tokens) > self.truncation_max_tokens:
                logger.info(f"Initial prompt too long ({len(initial_tokens)} tokens), applying truncation")
                messages = await self._apply_truncation(messages)
                conversation_messages = copy.deepcopy(messages)  # Update tracked messages
        
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    # tools=self.tool_schemas,  # TOREVIEW (Shankha): No tool schemas needed
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    # tools=self.tool_schemas,  # TOREVIEW (Shankha): No tool schemas needed
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        response_mask, response_logprobs = [], []
        # tools_kwargs = kwargs.get("tools_kwargs", {})  # TOREVIEW (Shankha): Not using tools_kwargs

        user_turns, assistant_turns = 0, 0
        # TOREVIEW (Shankha): Re-open the client for the main loop
        async with httpx.AsyncClient(timeout=self.modal_timeout) as client:
            
            # TOREVIEW (Jeffrey): Process initial assistant messages to extract and execute code
            # Similar to leader_agent.py lines 507-531
            logger.info("Processing initial assistant messages for code execution...")
            logger.info(f"Total initial messages: {len(conversation_messages)}")
            for i, msg in enumerate(conversation_messages):
                logger.info(f"Message {i}: role='{msg.get('role')}', content_length={len(msg.get('content', ''))}")
                if msg.get('role') == 'assistant' or (i == 0):
                    logger.info(f"Processing message {i} for code extraction (role: {msg.get('role')})")
                    # Extract code from assistant message content
                    code_blocks = self.extract_code_from_response(msg.get('content', ''))
                    logger.info(f"Extracted {len(code_blocks)} code blocks from message {i}")
                    if code_blocks:
                        logger.info(f"Found {len(code_blocks)} code block(s) in initial assistant message {i}, executing...")
                        for j, code_block in enumerate(code_blocks):
                            try:
                                # Execute code via Modal API
                                exec_response = await client.post(
                                    self._endpoint("execute-cell"),
                                    json={
                                        "instance_id": instance_id,
                                        "run_id": run_id,
                                        "notebook_id": notebook_id,
                                        "cell_content": code_block.strip()
                                    }
                                )
                                exec_data = exec_response.json()
                                
                                logger.info(f"Initial message {i} code block {j+1} execution result: "
                                          f"Success: {exec_data.get('success')}")
                                if exec_data.get('stdout'):
                                    logger.info(f"STDOUT: {exec_data['stdout'][:500]}...")  # Log first 500 chars
                                if exec_data.get('stderr'):
                                    logger.info(f"STDERR: {exec_data['stderr'][:500]}...")  # Log first 500 chars
                                    
                            except Exception as e:
                                logger.error(f"Failed to execute code from initial assistant message {i}, block {j+1}: {e}")
                    else:
                        logger.info(f"No code found in initial assistant message {i}")
            
            while True:
                # TOREVIEW (Shankha): Apply truncation before generation if enabled
                # This ensures we don't exceed context limits during training
                if self.enable_truncation and len(prompt_ids) > self.truncation_max_tokens:
                    logger.info(f"Prompt length {len(prompt_ids)} exceeds truncation threshold, applying truncation")
                    
                    # Apply truncation to conversation messages
                    truncated_messages = await self._apply_truncation(conversation_messages)
                    
                    # Re-tokenize the truncated conversation
                    # TOREVIEW (Jeffrey): Added lru_cache for tokenization caching
                    if self.processor is not None:
                        if self.enable_tokenization_cache and self._cached_apply_chat_template:
                            # caching on
                            messages_tuple = tuple(truncated_messages)
                            kwargs_tuple = tuple(sorted(self.apply_chat_template_kwargs.items()))
                            raw_prompt = await self.loop.run_in_executor(
                                None,
                                lambda: self._cached_apply_chat_template(messages_tuple, kwargs_tuple)
                            )
                        else:
                            # caching off
                            raw_prompt = await self.loop.run_in_executor(
                                None,
                                lambda: self.processor.apply_chat_template(
                                    truncated_messages,
                                    add_generation_prompt=True,
                                    tokenize=False,
                                    **self.apply_chat_template_kwargs,
                                ),
                            )
                    else:
                        prompt_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                truncated_messages,
                                add_generation_prompt=True,
                                tokenize=True,
                                **self.apply_chat_template_kwargs,
                            ),
                        )
                    
                    # Update conversation messages to reflect truncation
                    conversation_messages = truncated_messages
                    
                    # TOREVIEW (Shankha): CRITICAL - Response mask needs to be rebuilt after truncation
                    # TODO (Shankha): This is complex - we need to track which parts of the truncated
                    # conversation correspond to assistant vs tool responses
                    # For now, we'll have to regenerate from this point
                    logger.warning("Truncation applied - response mask and log probs will be regenerated from this point")
                    response_mask = []
                    response_logprobs = []
                    
                    # TOREVIEW (Shankha): After truncation, the model will regenerate responses
                    # with the compacted context, so log_probs will be consistent with the
                    # truncated prompt. This is the key insight from multi_turn_data_storage_correction.md
                
                with simple_timer("generate_sequences", metrics):
                    if self.use_modal_endpoint:
                        # Use Modal endpoint for generation
                        output = await self._generate_with_modal_endpoint(
                            conversation_messages, sampling_params, request_id
                        )
                    else:
                        # Use traditional server_manager
                        output = await self.server_manager.generate(
                            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                        )
                        # For server_manager, we need to decode the tokens to get the text output
                        if output.token_ids:
                            response_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                            save_llm_interaction(conversation_messages, response_text)
                response_ids = output.token_ids
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                if output.log_probs:
                    response_logprobs += output.log_probs
                assistant_turns += 1

                # reach max total turns (50) - only stop early if tool_calls == [] is detected in execution
                total_turns = assistant_turns
                if total_turns >= 50:
                    logger.info(f"Reached maximum turns limit (50): assistant_turns={assistant_turns}, user_turns={user_turns}")
                    break

                # TOREVIEW (Shankha): Extract code blocks instead of tool calls
                # Key insight: We're already converting tokens to text here for code extraction
                # So we can easily maintain conversation messages alongside token IDs
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # TOREVIEW (Shankha): Update conversation messages with assistant response
                conversation_messages.append({"role": "assistant", "content": response_text})
                
                code_blocks = self.extract_code_from_response(response_text)
                if not code_blocks:
                    # No code to execute - break the loop
                    logger.info("No code blocks found in response, ending loop")
                    break
                else:
                    # TOREVIEW (Shankha): Execute code blocks (sequentially, as notebooks are stateful)
                    tool_responses = []
                    execution_errors = []  # Track actual exceptions
                    with simple_timer("code_execution", metrics):
                        for code_block in code_blocks:
                            try:
                                # TOREVIEW (Shankha): Execute code via Modal API
                                exec_response = await client.post(
                                    self._endpoint("execute-cell"),  # Modal function: execute_cell
                                    json={
                                        "instance_id": instance_id,
                                        "run_id": run_id,
                                        "notebook_id": notebook_id,
                                        "cell_content": code_block.strip()
                                    }
                                )
                                exec_data = exec_response.json()
                            
                                # TOREVIEW (Jeffrey): Check for critical errors that should stop execution
                                if exec_data.get("terminated", False):
                                    # Kernel died or critical error - stop execution
                                    logger.error(f"Kernel terminated during execution")
                                    execution_errors.append(Exception("Kernel terminated"))
                                    tool_responses.append(ToolResponse(text="Error: Kernel terminated during execution"))
                                    break
                                
                                # TOREVIEW (Jeffrey): Check execution result for termination condition (tool_calls == [])
                                execution_model_output = self._extract_model_output_from_execution(exec_data)
                                if execution_model_output and execution_model_output.get("tool_calls") == []:
                                    logger.info(f"Execution result returned empty tool_calls, ending loop")
                                    # Signal completion by setting a flag that will be checked after the code execution loop
                                    execution_errors.append("COMPLETION_SIGNAL")
                                    tool_responses.append(ToolResponse(text=f"Task completed: {execution_model_output}"))
                                    break
                                
                                # TOREVIEW (Shankha): Create a ToolResponse compatible object
                                if exec_data.get("success"):
                                    output_text = ""
                                    if exec_data.get("stdout"):
                                        output_text += exec_data["stdout"]
                                    if exec_data.get("stderr"):
                                        output_text += f"\nSTDERR:\n{exec_data['stderr']}"
                                
                                    # TOREVIEW (Shankha): Apply truncation if output is too long (matching original tool_agent_loop behavior)
                                    # TODO: Consider implementing custom truncation logic for notebook outputs
                                    # For example: prioritize keeping error messages, truncate repetitive outputs differently,
                                    # or implement smart truncation that preserves data structure boundaries
                                    if output_text and len(output_text) > self.max_tool_response_length:
                                        if self.tool_response_truncate_side == "left":
                                            output_text = output_text[: self.max_tool_response_length] + "...(truncated)"
                                        elif self.tool_response_truncate_side == "right":
                                            output_text = "(truncated)..." + output_text[-self.max_tool_response_length:]
                                        else:  # middle truncation
                                            length = self.max_tool_response_length // 2
                                            output_text = output_text[:length] + "...(truncated)..." + output_text[-length:]
                                
                                    # Only create tool response if there's meaningful output
                                    if output_text and output_text.strip():
                                        tool_response = ToolResponse(text=output_text.strip())
                                        tool_responses.append(tool_response)
                                    # If no output, don't add any message to continue conversation
                                else:
                                    error_msg = exec_data.get("error", "Unknown execution error")
                                    tool_response = ToolResponse(text=f"Error: {error_msg}")
                                    tool_responses.append(tool_response)
                            
                            except Exception as e:
                                logger.error(f"Error executing code block: {e}")
                                execution_errors.append(e)
                                tool_responses.append(ToolResponse(text=f"Error executing code: {str(e)}"))
                            
                # TOREVIEW (Shankha): Break on critical execution errors or completion signal
                if execution_errors:
                    # Check if this is a completion signal (tool_calls == [])
                    if "COMPLETION_SIGNAL" in execution_errors:
                        logger.info(f"Breaking due to completion signal (tool_calls == [])")
                        break
                    else:
                        logger.warning(f"Breaking due to {len(execution_errors)} execution errors")
                        break

                # TOREVIEW (Shankha): Format tool responses as messages
                tool_messages = []
                # new_images_this_turn = []  # TOREVIEW (Shankha): Not handling images from code execution yet
                for tool_response in tool_responses:
                    # TOREVIEW (Shankha): Simplified message format for code outputs
                    message = {"role": "user", "content": tool_response.text or ""}
                    tool_messages.append(message)
                
                # TOREVIEW (Shankha): Update conversation messages with tool responses
                conversation_messages.extend(tool_messages)
                
                # Skip tokenization if no tool messages were generated
                if not tool_messages:
                    user_turns += 1
                    continue
                    
                    # TOREVIEW (Shankha): Image/video handling commented out for now
                    # if tool_response.image or tool_response.video:
                    #     # Multi-modal content with structured format
                    #     content = []
                    #     if tool_response.image:
                    #         content.append({"type": "image"})
                    #     if tool_response.video:
                    #         content.append({"type": "video"})
                    #     if tool_response.text:
                    #         content.append({"type": "text", "text": tool_response.text})
                    #     message = {"role": "tool", "content": content}
                    # else:
                    #     # Text-only content
                    #     message = {"role": "tool", "content": tool_response.text or ""}
                    
                    # tool_messages.append(message)
                    
                    # # Handle image data
                    # if tool_response.image:
                    #     if image_data is None:
                    #         image_data = []
                    #     elif not isinstance(image_data, list):
                    #         image_data = [image_data]
                    
                    #     # Add new image data
                    #     if isinstance(tool_response.image, list):
                    #         image_data.extend(tool_response.image)
                    #         new_images_this_turn.extend(tool_response.image)
                    #     else:
                    #         image_data.append(tool_response.image)
                    #         new_images_this_turn.append(tool_response.image)
                    
                    # # Handle video data
                    # if tool_response.video:
                    #     # Currently not supported, raise informative error
                    #     logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                    #     raise NotImplementedError(
                    #         "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    #     )

                # TOREVIEW (Shankha): Tokenize tool responses for appending to conversation
                if self.processor is not None:
                    raw_tool_response = await self.loop.run_in_executor(
                        None,
                        lambda messages=tool_messages: self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                        ),
                    )
                    # TOREVIEW (Shankha): No images from code execution currently
                    # current_images = new_images_this_turn if new_images_this_turn else None
                    model_inputs = self.processor(text=[raw_tool_response], images=None, return_tensors="pt")
                    tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                else:
                    tool_response_ids = await self.loop.run_in_executor(
                        None,
                        lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                        ),
                    )
                tool_response_ids = tool_response_ids[len(self.system_prompt) :]

                # NOTE: last turn should not be user turn, or the EOS token reward
                # can't be propagated to previous token in GAE.
                # Removed response length limit - only stop on meaningful conditions

                prompt_ids += tool_response_ids
                response_mask += [0] * len(tool_response_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(tool_response_ids)
                user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        # TOREVIEW (Shankha): Extract solution and terminate sandbox
        # Step 1: Get the solution patch for reward computation
        # Step 2: Terminate the sandbox to free resources
        # These are separate endpoints for clean separation of concerns
        solution_patch = ""
        solution_metadata = {}
        
        async with httpx.AsyncClient(timeout=self.modal_timeout) as client:
            # Step 1: Get solution patch
            try:
                # Call Modal endpoint to get solution patch
                # This endpoint returns the git diff of changes made in the sandbox
                solution_response = await client.post(
                    self._endpoint("get-solution-patch"),  # Modal function: get_solution_patch
                    json={
                        "instance_id": instance_id,
                        "run_id": run_id,
                        "notebook_id": notebook_id
                    }
                )
                
                if solution_response.status_code == 200:
                    solution_data = solution_response.json()
                    if solution_data.get("success"):
                        solution_patch = solution_data.get("patch", "")
                        solution_metadata = solution_data.get("metadata", {})
                        logger.info(f"Successfully extracted solution patch for {instance_id}")
                    else:
                        logger.warning(f"Failed to get solution: {solution_data}")
                else:
                    logger.error(f"Solution extraction failed with status {solution_response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error extracting solution: {e}")
                # Continue with empty solution patch - reward computation will handle this
            
            # Step 2: Terminate sandbox (separate from solution extraction)
            try:
                terminate_response = await client.post(
                    self._endpoint("terminate-sandbox"),  # Modal function: terminate_sandbox
                    json={
                        "instance_id": instance_id,
                        "run_id": run_id,
                        "notebook_id": notebook_id
                    }
                )
                if terminate_response.status_code == 200:
                    logger.info(f"Successfully terminated sandbox for {instance_id}")
                else:
                    logger.warning(f"Sandbox termination returned status {terminate_response.status_code}")
            except Exception as e:
                logger.error(f"Error terminating sandbox: {e}")
                # Non-critical error - sandbox will eventually timeout
        
        # TOREVIEW (Shankha): Include solution patch in metrics for reward computation
        # TODO (Shankha): Consider if this is the best way to pass the solution
        # Alternative: Add to non_tensor_batch in DataProto later in the pipeline
        metrics["solution_patch"] = solution_patch
        metrics["solution_metadata"] = solution_metadata
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    # TOREVIEW (Shankha): _call_tool method commented out - not using tool infrastructure
    # async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
    #     """Call tool and return tool response."""
    #     tool, instance_id = None, None
    #     try:
    #         # TODO: append malformed tool_call to the prompt: invalid function name or arguments
    #         tool_name = tool_call.name
    #         tool_args = json.loads(tool_call.arguments)
    #         tool = self.tools[tool_name]
    #         kwargs = tools_kwargs.get(tool_name, {})
    #         instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
    #         tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
    #     except Exception as e:
    #         logger.warning(f"Error when executing tool: {e}")
    #         return ToolResponse(
    #             text=f"Error when executing tool: {e}",
    #         )
    #     finally:
    #         if tool and instance_id:
    #             await tool.release(instance_id)
    
    #     tool_response_text = tool_execution_response.text
    #     if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
    #         if self.tool_response_truncate_side == "left":
    #             tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
    #         elif self.tool_response_truncate_side == "right":
    #             tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
    #         else:
    #             length = self.max_tool_response_length // 2
    #             tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]
    
    #     # Create ToolResponse from tool execution result
    #     tool_response_kwargs = {"text": tool_response_text}
    
    #     # Add multimedia data if present
    #     for attr_name in ["image", "video"]:
    #         if hasattr(tool_execution_response, attr_name):
    #             attr_value = getattr(tool_execution_response, attr_name)
    #             if attr_value is not None:
    #                 tool_response_kwargs[attr_name] = attr_value
    
    #     return ToolResponse(**tool_response_kwargs)
    
    # TOREVIEW (Shankha): Helper method to tokenize messages using existing VERL approach
    async def _tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Tokenize messages using the same approach as the main loop
        
        This avoids duplicating tokenize_conversation from compact_truncate.py
        and ensures we use VERL's tokenization (with processor support).
        """
        # Use the existing tokenization logic that handles processor vs tokenizer
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,  # Don't add generation prompt for length checking
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # For length checking, we just need the tokenized version without images
            tokens = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(raw_prompt)
            )
            return tokens
        else:
            tokens = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,  # Don't add generation prompt for length checking
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            return tokens
    
    # TOREVIEW (Shankha): Helper method to apply truncation following leader_agent.py pattern
    async def _apply_truncation(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Apply truncation to messages if enabled and needed"""
        if not self.enable_truncation or not self.truncation_strategy:
            return messages
        
        try:
            # Check if truncation is needed by tokenizing
            input_ids = await self._tokenize_messages(messages)
            current_tokens = len(input_ids)
            
            if current_tokens <= self.truncation_max_tokens:
                logger.debug(f"No truncation needed: {current_tokens} tokens <= {self.truncation_max_tokens}")
                return messages
            
            logger.info(f"Applying truncation: {current_tokens} tokens > {self.truncation_max_tokens}, strategy: {self.truncation_strategy}")
            
            # TOREVIEW (Shankha): Using the same truncation logic as leader_agent.py
            # IMPORTANT: Tokenization mismatch analysis:
            # 1. "first_user_priority" strategy uses tokenize_conversation which doesn't match VERL's tokenization
            # 2. "ast_llm_compaction" strategy doesn't use tokenization at all - it uses LLM summarization
            # 3. This mismatch is acceptable because:
            #    - Token counts are only used for approximate length checks
            #    - The actual training tokenization uses VERL's approach (with processor if available)
            #    - AST truncation produces semantically equivalent but shorter messages
            truncated_messages = truncate_conversation(
                messages,
                self.tokenizer,
                strategy=self.truncation_strategy,
                max_tokens=self.truncation_max_tokens,
                is_subleader=False  # TODO (Shankha): Determine if we need subleader logic here
            )
            
            # Verify truncation worked
            truncated_ids = await self._tokenize_messages(truncated_messages)
            logger.info(f"Truncated from {current_tokens} to {len(truncated_ids)} tokens")
            
            return truncated_messages
            
        except Exception as e:
            logger.error(f"Error during truncation: {e}")
            # TODO (Shankha): Should we fail or continue with original messages?
            # For now, continue with original to avoid breaking training
            return messages