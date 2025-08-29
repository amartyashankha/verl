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
from typing import Any
from uuid import uuid4

# MODAL: Import httpx for Modal API calls
import httpx
    
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
# from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser  # MODAL: Commented out - not using tool parsing
from verl.tools.schemas import ToolResponse
# from verl.tools.utils.tool_registry import initialize_tools_from_config  # MODAL: Commented out - not using tool registry
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("orchestrator_coding_agent")
class OrchestratorCodingAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level OrchestratorCodingAgentLoop initialization")

        # MODAL: Initialize basic attributes
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        # cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls  # MODAL: Not needed for sequential notebook execution
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length  # MODAL: Reuse for output length limit
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side  # MODAL: Reuse for output truncation
        
        # MODAL: Tool initialization commented out - using Modal API instead
        # tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        # tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        # cls.tools = {tool.name: tool for tool in tool_list}
        # cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        # cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        # print(f"Initialized tools: {cls.tools}")
        
        # MODAL: Initialize Modal-specific configuration
        cls.modal_base_url = config.actor_rollout_ref.rollout.multi_turn.get("modal_base_url", "https://fairies--incremental-leader-agent-api")
        cls.modal_timeout = config.actor_rollout_ref.rollout.multi_turn.get("modal_timeout", 300)
        cls.code_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)  # MODAL: Pattern to extract code blocks
        print(f"Initialized Modal agent with base URL: {cls.modal_base_url}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        
        # MODAL: Extract instance_id and run_id from kwargs
        instance_id = kwargs.get("instance_id", "default_instance")
        run_id = kwargs.get("run_id", f"run_{request_id}")
        notebook_id = kwargs.get("notebook_id", "main")
        
        # MODAL: Check if httpx is available
        if httpx is None:
            raise ImportError("httpx is required for Modal notebook agent. Install it with: pip install httpx")
        
        # MODAL: Initialize HTTP client and sandbox
        async with httpx.AsyncClient(timeout=self.modal_timeout) as client:
            # MODAL: Initialize the sandbox for this agent
            init_response = await client.post(
                f"{self.modal_base_url}-init-sandbox.modal.run",
                json={
                    "instance_id": instance_id,
                    "run_id": run_id,
                    "notebook_id": notebook_id,
                    "model_endpoint": "verl"  # TODO: Get from config if needed
                }
            )
            init_data = init_response.json()
            if init_data.get("status") != "success":
                logger.error(f"Failed to initialize Modal sandbox: {init_data}")
                # TODO: Decide how to handle initialization failure
                raise RuntimeError(f"Failed to initialize Modal sandbox: {init_data}")
            
            sandbox_id = init_data.get("sandbox_id")
            logger.info(f"Initialized Modal sandbox: {sandbox_id}")
        
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    # tools=self.tool_schemas,  # MODAL: No tool schemas needed
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
                    # tools=self.tool_schemas,  # MODAL: No tool schemas needed
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        response_mask, response_logprobs = [], []
        # tools_kwargs = kwargs.get("tools_kwargs", {})  # MODAL: Not using tools_kwargs

        user_turns, assistant_turns = 0, 0
        # MODAL: Re-open the client for the main loop
        async with httpx.AsyncClient(timeout=self.modal_timeout) as client:
            while True:
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                    )
                response_ids = output.token_ids
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)
                if output.log_probs:
                    response_logprobs += output.log_probs
                assistant_turns += 1

                # reach max response length
                if len(response_mask) >= self.response_length:
                    break

                # reach max assistant turns
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break

                # reach max user turns
                if self.max_user_turns and user_turns >= self.max_user_turns:
                    break

                # MODAL: Extract code blocks instead of tool calls
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                code_blocks = self.code_pattern.findall(response_text)
                if not code_blocks:
                    break  # No code to execute

                # MODAL: Execute code blocks (sequentially, as notebooks are stateful)
                tool_responses = []
                with simple_timer("code_execution", metrics):
                    for code_block in code_blocks:
                        try:
                            # MODAL: Execute code via Modal API
                            exec_response = await client.post(
                                f"{self.modal_base_url}-execute-cell.modal.run",
                                json={
                                    "instance_id": instance_id,
                                    "run_id": run_id,
                                    "notebook_id": notebook_id,
                                    "cell_content": code_block.strip()
                                }
                            )
                            exec_data = exec_response.json()
                            
                            # MODAL: Create a ToolResponse compatible object
                            if exec_data.get("success"):
                                output_text = ""
                                if exec_data.get("stdout"):
                                    output_text += exec_data["stdout"]
                                if exec_data.get("stderr"):
                                    output_text += f"\nSTDERR:\n{exec_data['stderr']}"
                                
                                # MODAL: Apply truncation if output is too long (matching original tool_agent_loop behavior)
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
                                
                                tool_response = ToolResponse(text=output_text.strip() if output_text else "Code executed successfully with no output.")
                            else:
                                error_msg = exec_data.get("error", "Unknown execution error")
                                tool_response = ToolResponse(text=f"Error: {error_msg}")
                            
                            tool_responses.append(tool_response)
                            
                        except Exception as e:
                            logger.error(f"Error executing code block: {e}")
                            tool_responses.append(ToolResponse(text=f"Error executing code: {str(e)}"))
                            
                if any(isinstance(item, Exception) for item in tool_responses):
                    break

                # MODAL: Format tool responses as messages
                tool_messages = []
                # new_images_this_turn = []  # MODAL: Not handling images from code execution yet
                for tool_response in tool_responses:
                    # MODAL: Simplified message format for code outputs
                    message = {"role": "tool", "content": tool_response.text or ""}
                    tool_messages.append(message)
                    
                    # MODAL: Image/video handling commented out for now
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

                # MODAL: Tokenize tool responses for appending to conversation
                if self.processor is not None:
                    raw_tool_response = await self.loop.run_in_executor(
                        None,
                        lambda messages=tool_messages: self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                        ),
                    )
                    # MODAL: No images from code execution currently
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
                if len(response_mask) + len(tool_response_ids) >= self.response_length:
                    break

                prompt_ids += tool_response_ids
                response_mask += [0] * len(tool_response_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(tool_response_ids)
                user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        # MODAL: TODO - Consider terminating sandbox at the end
        # We might want to keep it alive for the entire training episode
        # or terminate it here to free resources
        # Example:
        # await client.post(
        #     f"{self.modal_base_url}-terminate-sandbox.modal.run",
        #     json={
        #         "instance_id": instance_id,
        #         "run_id": run_id,
        #         "notebook_id": notebook_id
        #     }
        # )
        
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

    # MODAL: _call_tool method commented out - not using tool infrastructure
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
