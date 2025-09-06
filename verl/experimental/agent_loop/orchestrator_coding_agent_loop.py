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
import time
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

# Set logger level based on environment variable
log_level = os.getenv("VERL_LOGGING_LEVEL", "INFO")
if log_level.upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif log_level.upper() == "WARNING":
    logger.setLevel(logging.WARNING)
elif log_level.upper() == "ERROR":
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.INFO)

# Ensure logger outputs to console
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logger.level)  # Use the same level as the logger
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
            # Create timestamped filename in ~/RLlog/ - once per run
            timestamp = int(time.time())
            log_dir = os.path.expanduser("~/RLlog")
            os.makedirs(log_dir, exist_ok=True)
            _current_log_file = os.path.join(log_dir, f"test_llm_{timestamp}.json")
        log_file_path = _current_log_file
    
    try:
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
        
        # NEW: Oracle generation configuration
        cls.use_oracle_generation = config.actor_rollout_ref.rollout.multi_turn.get("use_oracle_generation", False)
        cls.oracle_messages_file = config.actor_rollout_ref.rollout.multi_turn.get(
            "oracle_messages_file",
            "/home/tianhangzhu/gcs_view/home/tianhangzhu/verltrain/claude-code_v2/noninteractive_results_swe_gym_train_rest/bokeh__bokeh-12779/ipynbs_subleader/SWEB_2025-07-31T03-31-27-463Z_main_messages.json"
        )
        # Optional: directory containing many oracle files, organized by instance_id subfolders
        cls.oracle_messages_dir = config.actor_rollout_ref.rollout.multi_turn.get("oracle_messages_dir", None)
        cls.oracle_messages = None
        cls.oracle_assistant_index = 0
        cls.oracle_user_index = 0
        cls._last_loaded_oracle_file = None  # Track the loaded oracle file path for finding solution.patch
        
        print(f"Initialized Modal agent with base URL: {cls.modal_base_url}")
        if cls.use_oracle_generation:
            # If a specific file is configured and exists, load immediately; otherwise defer to per-instance resolution
            if cls.oracle_messages_file and os.path.isfile(cls.oracle_messages_file):
                print(f"Using Oracle generation from: {cls.oracle_messages_file}")
                cls._load_oracle_messages()
            elif cls.oracle_messages_dir:
                print(f"Oracle generation enabled with directory: {cls.oracle_messages_dir} (will resolve per instance)")
            else:
                print("Oracle generation enabled but no file/dir configured; will attempt runtime resolution")
        elif cls.use_modal_endpoint:
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
        cls.enable_truncation = True
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        # Get prompt and response lengths from config
        cls.prompt_length = getattr(config.actor_rollout_ref.rollout, 'prompt_length', 
                                    getattr(config.data, 'max_prompt_length', 4096))
        cls.response_length = getattr(config.actor_rollout_ref.rollout, 'response_length',
                                      getattr(config.data, 'max_response_length', 2000))
        
        logger.info(f"Configured prompt_length: {cls.prompt_length}, response_length: {cls.response_length}")
        # Build a stable prefix token sequence for later offsetting tool response ids.
        # Some chat templates require a valid message with a role.
        try:
            minimal_messages = [{"role": "user", "content": ""}]
            if processor is not None:
                raw = processor.apply_chat_template(
                    minimal_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **cls.apply_chat_template_kwargs,
                )
                model_inputs = processor(text=[raw], images=None, return_tensors="pt")
                cls.system_prompt = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                cls.system_prompt = tokenizer.apply_chat_template(
                    minimal_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **cls.apply_chat_template_kwargs,
                )
        except Exception as e:
            logger.warning(f"Failed to build system_prompt via chat template: {e}; defaulting to empty prefix")
            cls.system_prompt = []

        # TOREVIEW (Jeffrey) Initialize LRU cache for tokenization
        cls.enable_tokenization_cache = config.actor_rollout_ref.rollout.multi_turn.get("enable_tokenization_cache", True)
        cls._cached_apply_chat_template = lru_cache(maxsize=1000)( # could probably make maxsize larger if you wanted
            cls._apply_chat_template_worker
        )
    
    @classmethod
    def _load_oracle_messages(cls):
        """Load oracle messages from JSON file"""
        try:
            with open(cls.oracle_messages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cls.oracle_messages = data.get("messages", [])
                cls.oracle_assistant_index = 0
                cls.oracle_user_index = 0
                print(f"✅ Loaded {len(cls.oracle_messages)} oracle messages")
                
                # Filter messages for replay
                assistant_messages = [msg for msg in cls.oracle_messages if msg.get("role") == "assistant"]
                user_messages = [msg for msg in cls.oracle_messages if msg.get("role") == "user"]
                print(f"✅ Found {len(assistant_messages)} assistant messages and {len(user_messages)} user messages to replay")
                
        except Exception as e:
            logger.error(f"Failed to load oracle messages from {cls.oracle_messages_file}: {e}")
            cls.oracle_messages = []
            cls.oracle_assistant_index = 0
            cls.oracle_user_index = 0

    @staticmethod
    @lru_cache(maxsize=2048)
    def _find_oracle_file(oracle_dir: str, instance_id: str) -> str | None:
        """Find oracle JSON file for a given instance_id under oracle_dir.
        
        Expected structure: {oracle_dir}/{instance_id}/ipynbs_subleader/SWEB_*_main_messages.json
        
        Returns the first matching file or None if not found.
        """
        if not oracle_dir or not os.path.isdir(oracle_dir) or not instance_id:
            return None
        
        logger.info(f"🔍 Looking for oracle file for instance: {instance_id}")
        logger.info(f"   Base directory: {oracle_dir}")
        
        # First, try the expected path structure
        expected_dir = os.path.join(oracle_dir, instance_id, "ipynbs_subleader")
        logger.info(f"   Checking expected directory: {expected_dir}")
        
        if os.path.isdir(expected_dir):
            logger.info(f"   ✅ Directory exists, looking for SWEB_*_main_messages.json files...")
            
            # List all files in the directory
            try:
                files = os.listdir(expected_dir)
                logger.info(f"   Found {len(files)} files in directory")
                
                # Look for SWEB_*_main_messages.json pattern
                sweb_files = []
                for fname in files:
                    if re.match(r"^SWEB_.*_main_messages\.json$", fname):
                        full_path = os.path.join(expected_dir, fname)
                        sweb_files.append(full_path)
                        logger.info(f"   ✅ Found SWEB oracle file: {fname}")
                
                if sweb_files:
                    # Return the first one (or could sort for consistency)
                    selected = sorted(sweb_files)[0]
                    logger.info(f"   📄 Selected oracle file: {os.path.basename(selected)}")
                    return selected
                else:
                    logger.warning(f"   ⚠️ No SWEB_*_main_messages.json files found in {expected_dir}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error reading directory {expected_dir}: {e}")
        else:
            logger.warning(f"   ⚠️ Expected directory does not exist: {expected_dir}")
        
        # Fallback: search recursively for any path containing instance_id
        logger.info(f"   Falling back to recursive search...")
        candidates = []
        
        try:
            for root, dirs, files in os.walk(oracle_dir):
                # Check if this directory is relevant to our instance
                if instance_id in root:
                    # Look for SWEB files in this directory
                    for fname in files:
                        if re.match(r"^SWEB_.*_main_messages\.json$", fname):
                            full_path = os.path.join(root, fname)
                            candidates.append(full_path)
                            logger.debug(f"   Found candidate: {full_path}")
                            
                            # Prefer files in ipynbs_subleader directories
                            if "ipynbs_subleader" in root:
                                logger.info(f"   ✅ Found SWEB file in ipynbs_subleader: {full_path}")
                                return full_path
                                
        except Exception as e:
            logger.error(f"   ❌ Error during recursive search: {e}")
        
        # Return the first candidate if any were found
        if candidates:
            selected = sorted(candidates)[0]
            logger.info(f"   📄 Selected oracle file from fallback search: {selected}")
            return selected
        
        logger.warning(f"⚠️ No oracle file found for instance_id: {instance_id}")
        return None

    def _ensure_oracle_loaded_for_instance(self, instance_id: str) -> None:
        """Ensure self.oracle_messages is loaded for the given instance_id.

        If a per-instance file is found under oracle_messages_dir, load it; otherwise
        fall back to class-configured oracle_messages_file (if any).
        """
        if not self.use_oracle_generation:
            return
        
        logger.info(f"🔮 Ensuring oracle loaded for instance: {instance_id}")
        
        # If already loaded for this instance, skip
        if getattr(self, "_loaded_oracle_instance_id", None) == instance_id and getattr(self, "oracle_messages", None):
            logger.info(f"✅ Oracle already loaded for {instance_id}: {len(self.oracle_messages)} messages")
            return
        
        selected_file = None
        if getattr(self, "oracle_messages_dir", None):
            logger.info(f"🔍 Searching for oracle file in directory: {self.oracle_messages_dir}")
            selected_file = self._find_oracle_file(self.oracle_messages_dir, instance_id)
            if selected_file:
                logger.info(f"✅ Found oracle file in directory: {selected_file}")
        
        if not selected_file and getattr(self, "oracle_messages_file", None) and os.path.isfile(self.oracle_messages_file):
            selected_file = self.oracle_messages_file
            logger.info(f"📄 Using configured oracle file: {selected_file}")
        
        if not selected_file:
            logger.warning(f"⚠️ Oracle messages file not found for instance_id={instance_id}")
            logger.warning(f"   Searched directory: {getattr(self, 'oracle_messages_dir', None)}")
            logger.warning(f"   Configured file: {getattr(self, 'oracle_messages_file', None)}")
            # Reset to empty to avoid crashes; generation will return empty
            self.oracle_messages = []
            self.oracle_assistant_index = 0
            self.oracle_user_index = 0
            self._loaded_oracle_instance_id = instance_id
            return
        
        # Load file
        try:
            logger.info(f"📥 Loading oracle file: {selected_file}")
            logger.info(f"   File size: {os.path.getsize(selected_file)} bytes")
            logger.info(f"   File name: {os.path.basename(selected_file)}")
            with open(selected_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.oracle_messages = data.get("messages", [])
                self.oracle_assistant_index = 0
                self.oracle_user_index = 0
                self._loaded_oracle_instance_id = instance_id
                self._last_loaded_oracle_file = selected_file  # Store the path for finding solution.patch
                
                # Log detailed breakdown
                assistant_msgs = [msg for msg in self.oracle_messages if msg.get("role") == "assistant"]
                user_msgs = [msg for msg in self.oracle_messages if msg.get("role") == "user"]
                system_msgs = [msg for msg in self.oracle_messages if msg.get("role") == "system"]
                
                logger.info(f"✅ Loaded oracle file for {instance_id}: {selected_file}")
                logger.info(f"   Total messages: {len(self.oracle_messages)}")
                logger.info(f"   Assistant messages: {len(assistant_msgs)}")
                logger.info(f"   User messages: {len(user_msgs)}")
                logger.info(f"   System messages: {len(system_msgs)}")
                
                # Log first and last assistant message previews
                if assistant_msgs:
                    first_content = assistant_msgs[0].get("content", "")
                    if isinstance(first_content, list):
                        first_content = str(first_content)
                    logger.info(f"   First assistant message preview: {first_content[:150]}...")
                    
                    last_content = assistant_msgs[-1].get("content", "")
                    if isinstance(last_content, list):
                        last_content = str(last_content)
                    logger.info(f"   Last assistant message preview: {last_content[:150]}...")
                    
        except Exception as e:
            logger.error(f"❌ Failed to load oracle file {selected_file} for {instance_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.oracle_messages = []
            self.oracle_assistant_index = 0
            self.oracle_user_index = 0
            self._loaded_oracle_instance_id = instance_id
    
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
                # Log the request details for debugging
                logger.info(f"Making Modal API request to: {self.modal_chat_endpoint}")
                logger.info(f"Request data: {json.dumps(request_data, indent=2)}")
                
                response = await client.post(
                    self.modal_chat_endpoint,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                logger.info(f"Modal API response status: {response.status_code}")
                logger.info(f"Modal API response headers: {dict(response.headers)}")
                
                # Capture response content BEFORE raise_for_status
                response_content = None
                try:
                    response_content = response.content.decode('utf-8')
                    logger.info(f"Modal API response content length: {len(response_content)} chars")
                except Exception as decode_err:
                    logger.error(f"Failed to decode response content: {decode_err}")
                    response_content = str(response.content)
                
                # Try to parse JSON response content before checking status
                response_json = None
                if response_content:
                    try:
                        response_json = json.loads(response_content)
                        logger.info(f"Modal API response JSON: {json.dumps(response_json, indent=2)}")
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Response content is not valid JSON: {json_err}")
                        logger.error(f"Raw response content: {response_content}")
                
                # Now check status and include response content in error
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code} from Modal API"
                    if response_json and 'error' in response_json:
                        error_msg += f": {response_json['error']}"
                    elif response_content:
                        error_msg += f": {response_content[:500]}..."
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
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
                
                # If it's an HTTP error, show detailed response information
                if hasattr(e, 'response'):
                    response = e.response
                    logger.error(f"HTTP Response Status: {response.status_code}")
                    logger.error(f"HTTP Response Headers: {dict(response.headers)}")
                    
                    # Capture and parse response content
                    try:
                        response_content = response.content.decode('utf-8')
                        logger.error(f"HTTP Response Content ({len(response_content)} chars): {response_content}")
                        
                        # Try to parse as JSON to get structured error
                        try:
                            error_json = json.loads(response_content)
                            logger.error(f"HTTP Response JSON Error: {json.dumps(error_json, indent=2)}")
                        except json.JSONDecodeError:
                            logger.error("HTTP Response is not valid JSON")
                            
                    except Exception as decode_err:
                        logger.error(f"Failed to decode response content: {decode_err}")
                        logger.error(f"HTTP Response Raw Content: {response.content}")
                
                # Fallback to empty response
                return TokenOutput(token_ids=[], log_probs=None)

    async def _generate_with_oracle(self, messages, sampling_params, request_id):
        """Generate text using oracle messages (replay assistant responses)"""
        from verl.workers.rollout.async_server import TokenOutput
        
        # Check if oracle messages are available
        if not self.oracle_messages:
            logger.warning("No oracle messages available - returning fallback response")
            # Return a minimal fallback response to avoid empty tensor issues
            fallback_text = "I cannot provide a solution as oracle data is not available for this task."
            response_ids = self.tokenizer.encode(fallback_text, add_special_tokens=False)
            # Use proper log probabilities (log space, not raw probabilities)
            import math
            log_probs = [math.log(0.5)] * len(response_ids) if response_ids else None
            return TokenOutput(token_ids=response_ids, log_probs=log_probs)
        
        # Find next assistant message to replay
        assistant_messages = [msg for msg in self.oracle_messages if msg.get("role") == "assistant"]
        
        if self.oracle_assistant_index >= len(assistant_messages):
            logger.warning(f"Oracle assistant index {self.oracle_assistant_index} exceeds available assistant messages ({len(assistant_messages)})")
            # Return a minimal fallback response instead of empty to avoid IndexError
            fallback_text = "Oracle messages exhausted - no more assistant responses available."
            response_ids = self.tokenizer.encode(fallback_text, add_special_tokens=False)
            # Use proper log probabilities (log space, not raw probabilities)
            import math
            log_probs = [math.log(0.5)] * len(response_ids) if response_ids else None
            return TokenOutput(token_ids=response_ids, log_probs=log_probs)
        
        # Get the current assistant message to replay
        current_message = assistant_messages[self.oracle_assistant_index]
        self.oracle_assistant_index += 1
        
        # Extract content from the message
        content = current_message.get("content", "")
        if isinstance(content, list):
            # Handle structured content (like from Claude API)
            text_content = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
                elif isinstance(item, str):
                    text_content += item
            content = text_content
        
        logger.info(f"Oracle replaying assistant message {self.oracle_assistant_index}/{len(assistant_messages)}")
        logger.info(f"Oracle content preview: {content[:200]}...")
        
        # Save the oracle interaction for logging
        save_llm_interaction(messages, content)
        
        # Tokenize the oracle response
        response_ids = self.tokenizer.encode(content, add_special_tokens=False)
        
        # Generate log probabilities in log space (not raw probabilities)
        # Using log(0.5) ≈ -0.693 as a placeholder for oracle responses
        # Note: These are dummy values since we're replaying oracle messages
        import math
        log_probs = [math.log(0.5)] * len(response_ids) if response_ids else None
        
        logger.info(f"Oracle generated {len(response_ids)} tokens with {len(log_probs) if log_probs else 0} log_probs")
        
        return TokenOutput(
            token_ids=response_ids,
            log_probs=log_probs
        )

    def _get_next_oracle_user_message(self):
        """Get the next user message from oracle for fake execution results"""
        user_messages = [msg for msg in self.oracle_messages if msg.get("role") == "user"]
        
        if self.oracle_user_index >= len(user_messages):
            logger.warning(f"Oracle user index {self.oracle_user_index} exceeds available user messages ({len(user_messages)})")
            return "Oracle user message not available"
        
        # Get the current user message to use as fake execution result
        current_message = user_messages[self.oracle_user_index]
        self.oracle_user_index += 1
        
        # Extract content from the message
        content = current_message.get("content", "")
        if isinstance(content, list):
            # Handle structured content (like from Claude API)
            text_content = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
                elif isinstance(item, str):
                    text_content += item
            content = text_content
        
        logger.info(f"Oracle using user message {self.oracle_user_index}/{len(user_messages)} as fake execution result")
        logger.info(f"Oracle user content preview: {content[:200]}...")
        
        return content

    def _extract_solution_from_oracle_messages(self):
        """Extract the solution patch from solution.patch file in the oracle directory"""
        logger.info(f"🔍 Looking for solution.patch file for instance: {getattr(self, '_loaded_oracle_instance_id', 'unknown')}")
        
        # First, try to find solution.patch file based on the oracle directory structure
        if hasattr(self, '_loaded_oracle_instance_id') and self._loaded_oracle_instance_id:
            instance_id = self._loaded_oracle_instance_id
            
            # Try different possible locations for solution.patch
            possible_paths = []
            
            # If we have oracle_messages_dir, look for solution.patch there
            if hasattr(self, 'oracle_messages_dir') and self.oracle_messages_dir:
                base_dir = self.oracle_messages_dir
                
                # Pattern 1: {base_dir}/noninteractive_results_swe_gym_train_rest/{instance_id}/solution.patch
                path1 = os.path.join(base_dir, "noninteractive_results_swe_gym_train_rest", instance_id, "solution.patch")
                possible_paths.append(path1)
                
                # Pattern 2: {base_dir}/{instance_id}/solution.patch
                path2 = os.path.join(base_dir, instance_id, "solution.patch")
                possible_paths.append(path2)
                
                # Pattern 3: Same directory as the oracle messages file
                if hasattr(self, '_last_loaded_oracle_file') and self._last_loaded_oracle_file:
                    oracle_dir = os.path.dirname(self._last_loaded_oracle_file)
                    # Go up one level from ipynbs_subleader to instance directory
                    if "ipynbs_subleader" in oracle_dir:
                        instance_dir = os.path.dirname(oracle_dir)
                        path3 = os.path.join(instance_dir, "solution.patch")
                        possible_paths.append(path3)
            
            # Try each possible path
            for patch_path in possible_paths:
                logger.info(f"   Checking for solution.patch at: {patch_path}")
                if os.path.exists(patch_path):
                    try:
                        with open(patch_path, 'r', encoding='utf-8') as f:
                            patch_content = f.read()
                        logger.info(f"✅ Found solution.patch file: {patch_path}")
                        logger.info(f"   Patch size: {len(patch_content)} characters")
                        logger.info(f"   Patch preview: {patch_content[:200]}...")
                        return patch_content
                    except Exception as e:
                        logger.error(f"❌ Error reading solution.patch from {patch_path}: {e}")
            
            logger.warning(f"⚠️ No solution.patch file found for instance {instance_id}")
            logger.warning(f"   Searched paths: {possible_paths}")
        
        # Fallback: Try to extract from oracle messages (original logic)
        logger.info("📝 Falling back to extracting solution from oracle messages...")
        
        if not self.oracle_messages:
            logger.error("❌ No oracle messages loaded!")
            return "No oracle messages loaded"
        
        # Log message types and counts
        assistant_msgs = [msg for msg in self.oracle_messages if msg.get("role") == "assistant"]
        user_msgs = [msg for msg in self.oracle_messages if msg.get("role") == "user"]
        logger.info(f"📊 Oracle message breakdown: {len(assistant_msgs)} assistant, {len(user_msgs)} user messages")
        
        # Look for messages that contain solution patches or git diffs
        for idx, message in enumerate(self.oracle_messages):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                
                # Handle structured content
                if isinstance(content, list):
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif isinstance(item, str):
                            text_content += item
                    content = text_content
                
                # Look for git diff patterns
                if content and ("diff --git" in content or content.startswith("diff")):
                    logger.info(f"✅ Found solution patch in oracle message {idx}: {len(content)} characters")
                    return content
                
                # Check for unified diff format
                if content and "\n@@" in content and ("\n+" in content or "\n-" in content):
                    logger.info(f"✅ Found unified diff format in message {idx}")
                    return content
        
        logger.warning("⚠️ No solution patch found in oracle messages or solution.patch file")
        return "No solution patch found"

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
        """
        CRITICAL PPO TRAINING NOTES (Jeffrey):
        
        1. response_mask and response_logprobs accumulate ALL assistant responses throughout
           the entire conversation. They are NEVER reset, even after truncation.
           
        2. Truncation only affects the conversation context used for the NEXT generation.
           It does NOT affect the accumulated training data.
           
        3. response_mask values:
           - 1 for assistant tokens (trainable)
           - 0 for tool/user response tokens (non-trainable)
           
        4. response_logprobs for non-trainable tokens are set to -inf to ensure they
           don't contribute to gradients during PPO training.
           
        5. The final output contains ALL tokens generated during the conversation,
           preserving the complete trajectory for PPO advantage computation.
        """
        instance_id = kwargs.get("instance_id", "default_instance")
        logger.info(f"🚀 OrchestratorCodingAgentLoop.run() started for instance: {instance_id}")
        logger.info(f"   Oracle generation enabled: {self.use_oracle_generation}")
        logger.info(f"   Modal endpoint enabled: {self.use_modal_endpoint}")
        
        # ORACLE MODE: Simplified flow - just run first cell and extract patch
        if self.use_oracle_generation:
            logger.info("🔮 Switching to ORACLE MODE for simplified execution")
            return await self._run_oracle_mode(sampling_params, **kwargs)
        
        # NORMAL MODE: Full prompt generation and execution
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
            # Initialize the sandbox for this agent
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
        # CRITICAL: These accumulate ALL responses throughout the conversation
        # They should NEVER be reset, even after truncation
        response_mask, response_logprobs = [], []
        # tools_kwargs = kwargs.get("tools_kwargs", {})  # TOREVIEW (Shankha): Not using tools_kwargs

        user_turns, assistant_turns = 0, 0
        # TOREVIEW (Shankha): Re-open the client for the main loop
        async with httpx.AsyncClient(timeout=self.modal_timeout) as client:
            
            # TOREVIEW (Jeffrey): Process initial assistant messages to extract and execute code
            # IMPORTANT: If there are initial assistant messages in the conversation,
            # we need to track them in response_mask and response_logprobs for training
            logger.info("Processing initial messages...")
            logger.info(f"Total initial messages: {len(conversation_messages)}")
            
            # Check if we have any initial assistant messages to process
            has_initial_assistant = any(msg.get('role') == 'assistant' for msg in conversation_messages)
            
            if has_initial_assistant:
                logger.info("Found initial assistant messages - tracking for training")
                # We need to tokenize and track these initial assistant messages
                # Note: This assumes the initial messages are already part of prompt_ids
                # and we need to extract which portions are assistant vs user
                
                # For now, just execute any code blocks found
                for i, msg in enumerate(conversation_messages):
                    logger.info(f"Message {i}: role='{msg.get('role')}', content_length={len(msg.get('content', ''))}")
                    if msg.get('role') == 'assistant':
                        logger.info(f"Processing assistant message {i} for code extraction")
                        
                        # TODO: We should tokenize this message and add to response tracking
                        # For now, just extract and execute code
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
                # IMPORTANT: Truncation only affects the context for next generation,
                # NOT the accumulated responses we're training on
                if self.enable_truncation:
                    # Check if current conversation context exceeds limits
                    context_tokens = await self._tokenize_messages(conversation_messages)
                    if len(context_tokens) > self.truncation_max_tokens:
                        logger.info(f"Context length {len(context_tokens)} exceeds truncation threshold, applying truncation")
                        
                        # Apply truncation to conversation messages for next generation
                        truncated_messages = await self._apply_truncation(conversation_messages)
                        
                        # Update conversation messages to reflect truncation
                        conversation_messages = truncated_messages
                        
                        # Re-tokenize the truncated conversation for next generation
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
                            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
                            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
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
                        
                        logger.info(f"Truncation applied to context, but preserving all {len(response_mask)} response tokens for training")
                        # CRITICAL: We do NOT reset response_mask or response_logprobs here!
                        # We continue accumulating all responses regardless of truncation
                
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
                                # Execute code via Modal API
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
                                        output_text += exec_data['stderr']
                                
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

                prompt_ids += tool_response_ids # interleaved, we only care about system messages here
                response_mask += [0] * len(tool_response_ids)
                # CRITICAL: Always maintain response_logprobs in sync with response_mask
                # Even if response_logprobs was initially empty, we need to keep them aligned
                if response_logprobs is not None:
                    # Use -inf for non-trainable tokens (tool responses)
                    # This ensures they don't contribute to gradients during PPO training
                    response_logprobs += [float('-inf')] * len(tool_response_ids)
                else:
                    # If we don't have log probs yet, initialize with -inf for tool responses
                    response_logprobs = [float('-inf')] * len(tool_response_ids)
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
            # Extract solution and terminate sandbox
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
        
        # Compute reward score directly here instead of passing through pipeline
        reward_score = 0.0
        if solution_patch:
            logger.info(f"[REWARD] Computing reward for {instance_id} with patch length {len(solution_patch)}")
            try:
                import httpx
                modal_evaluation_url = kwargs.get("modal_evaluation_url", "https://fairies--swe-gym-evaluation-service-polling-fastapi-app.modal.run")
                dataset_name = kwargs.get("dataset_name", "SWE-Gym/SWE-Gym")
                split = kwargs.get("split", "train")
                run_id = kwargs.get("run_id", f"verl_eval_{instance_id}")
                
                with httpx.Client(timeout=600) as client:
                    # Step 1: Submit the patch
                    logger.info(f"[REWARD] Submitting patch to {modal_evaluation_url}/submit")
                    submit_response = client.post(
                        f"{modal_evaluation_url}/submit",
                        json={
                            "instance_id": instance_id,
                            "patch": solution_patch,
                            "run_id": run_id
                        }
                    )
                    
                    if submit_response.status_code == 200:
                        submit_data = submit_response.json()
                        call_id = submit_data.get("call_id")
                        logger.info(f"[REWARD] Submitted successfully, call_id: {call_id}")
                        
                        # Step 2: Poll for result
                        result_url = f"{modal_evaluation_url}/result/{call_id}"
                        max_poll_time = 300  # 5 minutes max
                        poll_interval = 3  # seconds
                        start_time = time.time()
                        
                        while time.time() - start_time < max_poll_time:
                            result_response = client.get(result_url)
                            
                            if result_response.status_code == 200:
                                # Got result
                                result_data = result_response.json()
                                if result_data.get("success"):
                                    test_results = result_data.get("test_results", {})
                                    tests_status = test_results.get("tests_status", {})
                                    
                                    # Check if resolved (all tests pass)
                                    fail_to_pass = tests_status.get("FAIL_TO_PASS", {}).get("failure", [])
                                    pass_to_pass = tests_status.get("PASS_TO_PASS", {}).get("failure", [])
                                    fail_to_fail = tests_status.get("FAIL_TO_FAIL", {}).get("success", [])
                                    pass_to_fail = tests_status.get("PASS_TO_FAIL", {}).get("success", [])
                                    
                                    resolved = (
                                        len(fail_to_pass) == 0 and
                                        len(pass_to_pass) == 0 and
                                        len(fail_to_fail) == 0 and
                                        len(pass_to_fail) == 0
                                    )
                                    
                                    # Calculate partial score based on fail_to_pass
                                    fail_to_pass_success = tests_status.get("FAIL_TO_PASS", {}).get("success", [])
                                    fail_to_fail_failure = tests_status.get("FAIL_TO_FAIL", {}).get("failure", [])
                                    total_originally_failing = len(fail_to_pass_success) + len(fail_to_pass) + len(fail_to_fail_failure) + len(fail_to_fail)
                                    
                                    if resolved:
                                        reward_score = 1.0
                                    elif total_originally_failing > 0:
                                        reward_score = len(fail_to_pass_success) / total_originally_failing
                                    else:
                                        reward_score = 0.0
                                    
                                    logger.info(f"[REWARD] Score for {instance_id}: {reward_score:.3f} (resolved={resolved}, F2P={len(fail_to_pass_success)}/{total_originally_failing})")
                                else:
                                    logger.error(f"[REWARD] Evaluation failed: {result_data.get('error')}")
                                break
                                
                            elif result_response.status_code == 202:
                                # Still processing
                                logger.debug(f"[REWARD] Still processing {call_id}, polling again...")
                                time.sleep(poll_interval)
                                
                            elif result_response.status_code == 404:
                                logger.error(f"[REWARD] Result not found or expired for {call_id}")
                                break
                                
                            else:
                                logger.error(f"[REWARD] Unexpected status {result_response.status_code} when polling")
                                break
                        else:
                            logger.error(f"[REWARD] Polling timeout after {max_poll_time}s")
                    else:
                        logger.error(f"[REWARD] Submit failed with HTTP {submit_response.status_code}")
            except Exception as e:
                logger.error(f"[REWARD] Error computing reward: {e}")
        else:
            logger.info(f"[REWARD] No solution patch for {instance_id}, using reward=0.0")
        
        # Still include in metrics for logging
        metrics["solution_patch"] = solution_patch
        metrics["solution_metadata"] = solution_metadata
        metrics["reward_score"] = reward_score
        
        # Ensure prompt_ids don't exceed max length to avoid tensor size mismatch
        if len(prompt_ids) > self.prompt_length:
            logger.warning(f"Truncating prompt_ids from {len(prompt_ids)} to {self.prompt_length}")
            prompt_ids = prompt_ids[: self.prompt_length]
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            reward_score=reward_score,  # Pass the computed reward directly
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
    
    async def _run_oracle_mode(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Simplified oracle mode matching run_oracle_task.py flow"""
        logger.info("🔮 ENTERING ORACLE MODE")
        
        # Extract parameters
        instance_id = kwargs.get("instance_id", "default_instance")
        task_prompt = kwargs.get("task_prompt", "")
        
        logger.info(f"📋 Oracle Mode Parameters:")
        logger.info(f"   Instance ID: {instance_id}")
        logger.info(f"   Task prompt length: {len(task_prompt)} chars")
        logger.info(f"   Task prompt preview: {task_prompt[:200]}...")
        
        # Build first cell (same as run_oracle_task.py) - MUST happen before checking oracle messages
        full_prompt, first_cell_code = build_full_prompt_for_first_cell(task_prompt)
        logger.info(f"📝 Built first cell prompt: {len(full_prompt)} chars")
        
        # Load oracle messages for this instance
        self._ensure_oracle_loaded_for_instance(instance_id)
        
        # Check if oracle messages were loaded
        if not self.oracle_messages:
            logger.warning(f"⚠️ No oracle messages for {instance_id}, skipping (this is expected for new instances)")
            # Just return minimal valid output - don't try to generate anything
            return AgentLoopOutput(
                prompt_ids=[1],  # Minimal valid prompt
                response_ids=[1],  # Minimal valid response  
                response_mask=[1],  # Minimal valid mask
                multi_modal_data={},
                response_logprobs=[0.0],  # Minimal valid log prob
                num_turns=1,
                metrics={"skipped": True, "reason": "no_oracle", "solution_patch": "", "instance_id": instance_id},
            )
        
        logger.info(f"✅ Oracle messages loaded: {len(self.oracle_messages)} total messages")
        
        # Create initial messages for tokenization
        messages = [{"role": "user", "content": full_prompt}]
        
        # Tokenize the initial prompt to get prompt_ids
        logger.info("🔤 Tokenizing initial prompt...")
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
        logger.info(f"✅ Prompt tokenized: {len(prompt_ids)} tokens")
        
        # Generate oracle response (replay assistant message)
        logger.info("🎭 Generating oracle response (replaying assistant message)...")
        request_id = uuid4().hex
        output = await self._generate_with_oracle(messages, sampling_params, request_id)
        response_ids = output.token_ids
        response_logprobs = output.log_probs
        
        logger.info(f"✅ Oracle response generated:")
        logger.info(f"   Response tokens: {len(response_ids)}")
        logger.info(f"   Has log probs: {response_logprobs is not None}")
        if response_logprobs:
            logger.info(f"   Log probs length: {len(response_logprobs)}")
            logger.info(f"   Sample log probs: {response_logprobs[:5] if len(response_logprobs) > 5 else response_logprobs}")
        
        # Decode response to see what was generated
        if response_ids:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            logger.info(f"   Response text ({len(response_text)} chars): {response_text[:200]}...")
        
        # Build response mask
        response_mask = [1] * len(response_ids)
        logger.info(f"✅ Response mask built: {len(response_mask)} elements")
        
        # Extract solution patch from oracle messages
        logger.info("🔍 Extracting solution patch from oracle messages...")
        solution_patch = self._extract_solution_from_oracle_messages()
        
        # Compute reward score for oracle mode
        reward_score = 0.0
        if solution_patch and solution_patch not in ["No solution patch found in oracle messages", "No oracle messages loaded", "No solution patch found"]:
            logger.info(f"✅ Solution patch extracted: {len(solution_patch)} chars")
            logger.info(f"   Solution preview: {solution_patch[:200]}...")
            
            # Compute reward using the same logic as normal mode
            try:
                import httpx
                modal_evaluation_url = kwargs.get("modal_evaluation_url", "https://fairies--swe-gym-evaluation-service-polling-fastapi-app.modal.run")
                dataset_name = kwargs.get("dataset_name", "SWE-Gym/SWE-Gym")
                split = kwargs.get("split", "train")
                run_id = kwargs.get("run_id", f"verl_eval_{instance_id}")
                
                with httpx.Client(timeout=600) as client:
                    logger.info(f"[ORACLE REWARD] Submitting to {modal_evaluation_url}/submit")
                    submit_response = client.post(
                        f"{modal_evaluation_url}/submit",
                        json={
                            "instance_id": instance_id,
                            "patch": solution_patch,
                            "run_id": run_id
                        }
                    )
                    
                    if submit_response.status_code == 200:
                        submit_data = submit_response.json()
                        call_id = submit_data.get("call_id")
                        logger.info(f"[ORACLE REWARD] Submitted successfully, call_id: {call_id}")
                        
                        # Step 2: Poll for result
                        result_url = f"{modal_evaluation_url}/result/{call_id}"
                        max_poll_time = 300  # 5 minutes max
                        poll_interval = 3  # seconds
                        start_time = time.time()
                        
                        while time.time() - start_time < max_poll_time:
                            result_response = client.get(result_url)
                            
                            if result_response.status_code == 200:
                                # Got result
                                result_data = result_response.json()
                                if result_data.get("success"):
                                    test_results = result_data.get("test_results", {})
                                    tests_status = test_results.get("tests_status", {})
                                    
                                    # Check if resolved (all tests pass)
                                    fail_to_pass = tests_status.get("FAIL_TO_PASS", {}).get("failure", [])
                                    pass_to_pass = tests_status.get("PASS_TO_PASS", {}).get("failure", [])
                                    fail_to_fail = tests_status.get("FAIL_TO_FAIL", {}).get("success", [])
                                    pass_to_fail = tests_status.get("PASS_TO_FAIL", {}).get("success", [])
                                    
                                    resolved = (
                                        len(fail_to_pass) == 0 and
                                        len(pass_to_pass) == 0 and
                                        len(fail_to_fail) == 0 and
                                        len(pass_to_fail) == 0
                                    )
                                    
                                    # Calculate partial score based on fail_to_pass
                                    fail_to_pass_success = tests_status.get("FAIL_TO_PASS", {}).get("success", [])
                                    fail_to_fail_failure = tests_status.get("FAIL_TO_FAIL", {}).get("failure", [])
                                    total_originally_failing = len(fail_to_pass_success) + len(fail_to_pass) + len(fail_to_fail_failure) + len(fail_to_fail)
                                    
                                    if resolved:
                                        reward_score = 1.0
                                    elif total_originally_failing > 0:
                                        reward_score = len(fail_to_pass_success) / total_originally_failing
                                    else:
                                        reward_score = 0.0
                                    
                                    logger.info(f"[ORACLE REWARD] Score for {instance_id}: {reward_score:.3f} (resolved={resolved}, F2P={len(fail_to_pass_success)}/{total_originally_failing})")
                                else:
                                    logger.error(f"[ORACLE REWARD] Evaluation failed: {result_data.get('error')}")
                                break
                                
                            elif result_response.status_code == 202:
                                # Still processing
                                logger.debug(f"[ORACLE REWARD] Still processing {call_id}, polling again...")
                                time.sleep(poll_interval)
                                
                            elif result_response.status_code == 404:
                                logger.error(f"[ORACLE REWARD] Result not found or expired for {call_id}")
                                break
                                
                            else:
                                logger.error(f"[ORACLE REWARD] Unexpected status {result_response.status_code} when polling")
                                break
                        else:
                            logger.error(f"[ORACLE REWARD] Polling timeout after {max_poll_time}s")
                    else:
                        logger.error(f"[ORACLE REWARD] Submit failed with HTTP {submit_response.status_code}")
            except Exception as e:
                logger.error(f"[ORACLE REWARD] Error computing reward: {e}")
        else:
            logger.warning(f"⚠️ No valid solution patch found: {solution_patch}")
        
        # Build metrics
        metrics = {
            "solution_patch": solution_patch,
            "solution_metadata": {"oracle_mode": True, "instance_id": instance_id},
            "oracle_messages_count": len(self.oracle_messages) if self.oracle_messages else 0
        }
        
        logger.info(f"📊 Oracle Mode Metrics:")
        logger.info(f"   Oracle messages count: {metrics['oracle_messages_count']}")
        logger.info(f"   Solution patch length: {len(solution_patch) if solution_patch else 0}")
        logger.info(f"   Instance ID: {instance_id}")
        
        # Ensure prompt_ids don't exceed max length to avoid tensor size mismatch
        if len(prompt_ids) > self.prompt_length:
            logger.warning(f"[ORACLE] Truncating prompt_ids from {len(prompt_ids)} to {self.prompt_length}")
            prompt_ids = prompt_ids[: self.prompt_length]
        
        # Return simplified output
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data={},
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=1,  # Oracle mode is single turn
            metrics=metrics,
            reward_score=reward_score,  # Pass the computed reward directly
        )
        
        logger.info("🔮 ORACLE MODE COMPLETE")
        return output