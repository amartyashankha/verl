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
"""
Orchestrator Modal Notebook Agent Recipe
"""

import asyncio
import logging
import os
from typing import Union

import datasets
import numpy as np
import ray
import torch
from torch import nn
from torch.nn import functional as F

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score.grader import Grader

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class OrchestratorDataset(RLHFDataset):
    """Dataset for orchestrator notebook agent with Modal sandboxes."""
    
    def __init__(self, data_files, tokenizer, prompt_key="query", response_key="response", 
                 apply_chat_template=None, enable_truncation=False, truncation_strategy="ast_llm_compaction",
                 truncation_max_tokens=16000, **kwargs):
        """Initialize orchestrator dataset.
        
        Args:
            enable_truncation: Whether to enable conversation truncation during training
            truncation_strategy: Which truncation strategy to use
            truncation_max_tokens: Maximum tokens before truncation
        """
        self.enable_truncation = enable_truncation
        self.truncation_strategy = truncation_strategy
        self.truncation_max_tokens = truncation_max_tokens
        
        super().__init__(data_files, tokenizer, prompt_key, response_key, 
                         apply_chat_template, **kwargs)
    
    def _read_files_and_tokenize(self):
        # Load data files (could be from SWE-bench, custom datasets, etc.)
        dataframes = []
        for data_file in self.data_files:
            dataframe = datasets.load_dataset(data_file)["train"]
            # Process each row to add orchestrator-specific fields
            dataframe = dataframe.map(self.map_fn, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)
        logger.info(f"Loaded {len(self.dataframe)} examples for orchestrator agent")
    
    def map_fn(self, row: dict):
        """Map dataset row to orchestrator format."""
        # Extract problem/task description
        problem = row.get("problem", row.get("prompt", ""))
        instance_id = row.get("instance_id", f"instance_{row.get('id', 0)}")
        
        # Create initial prompt for the orchestrator
        prompt_content = f"""You are an AI coding assistant working in a Jupyter-like notebook environment.

Problem: {problem}

Instructions:
- Write Python code to solve this problem
- Each code block should be wrapped in <code>...</code> tags
- The code will be executed sequentially in a persistent environment
- You can use print statements to debug and verify your solution
- Import any necessary libraries at the beginning

Begin by understanding the problem and implementing a solution."""
        
        data = {
            "data_source": row.get("data_source", "orchestrator_dataset"),
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": row.get("ability", "CODING"),
            "reward_model": {
                "ground_truth": row.get("ground_truth", row.get("answer", "")),
                "test_cases": row.get("test_cases", []),
            },
            "agent_name": "orchestrator_coding_agent",  # CRITICAL: This selects our agent
            "instance_id": instance_id,
            "run_id": f"run_{instance_id}",
            "notebook_id": "main",
            # Pass truncation config to the agent
            "agent_config": {
                "enable_truncation": self.enable_truncation,
                "truncation_strategy": self.truncation_strategy,
                "truncation_max_tokens": self.truncation_max_tokens,
            }
        }
        return data


async def compute_score(data_dict: dict, response: Union[str, list[dict]], grader: Grader = None, **kwargs) -> float:
    """Compute reward score for orchestrator agent responses.
    
    This is called after the agent completes its notebook execution.
    The solution patch is extracted by the agent loop and passed through the pipeline.
    """
    import httpx
    
    # TODO (Shankha): Extract solution patch from kwargs
    # The solution patch is passed through the metrics in AgentLoopOutput
    extra_info = kwargs.get("extra_info", {})
    metrics = extra_info.get("metrics", {})
    solution_patch = metrics.get("solution_patch", "")
    solution_metadata = metrics.get("solution_metadata", {})
    
    # TODO (Shankha): Add logging to debug data flow
    logger.info(f"Computing reward for instance {data_dict.get('instance_id')}")
    logger.debug(f"Solution patch length: {len(solution_patch)}")
    logger.debug(f"Solution metadata: {solution_metadata}")
    
    if not solution_patch:
        logger.warning("No solution patch found in metrics, checking for fallback")
        # TODO (Shankha): Implement fallback logic if needed
        # For example, try to extract from response text
        return 0.0
    
    # TODO (Shankha): Get Modal endpoint URL from config
    # This should come from the orchestrator config, not hardcoded
    modal_reward_url = "https://fairies--incremental-leader-agent-api-compute-reward.modal.run"
    
    try:
        # TODO (Shankha): Call Modal reward computation endpoint
        async with httpx.AsyncClient(timeout=300) as client:
            reward_response = await client.post(
                modal_reward_url,
                json={
                    "instance_id": data_dict.get("instance_id"),
                    "solution_patch": solution_patch,
                    "ground_truth": data_dict.get("ground_truth", ""),
                    "test_cases": data_dict.get("test_cases", []),
                    # TODO (Shankha): Add additional context if needed
                    "metadata": solution_metadata,
                    "problem": data_dict.get("problem", ""),
                }
            )
            
            if reward_response.status_code == 200:
                reward_data = reward_response.json()
                score = reward_data.get("score", 0.0)
                
                # TODO (Shankha): Extract additional information from reward response
                # For example: test results, execution logs, etc.
                if "details" in reward_data:
                    logger.info(f"Reward details: {reward_data['details']}")
                
                # TODO (Shankha): Consider normalizing score to [0, 1] range
                # The Modal endpoint might return different score ranges
                score = max(0.0, min(1.0, score))
                
                return score
            else:
                logger.error(f"Reward computation failed with status {reward_response.status_code}")
                logger.error(f"Response: {reward_response.text}")
                return 0.0
                
    except httpx.TimeoutException:
        logger.error("Reward computation timed out")
        # TODO (Shankha): Implement retry logic or fallback scoring
        return 0.0
    except Exception as e:
        logger.error(f"Error computing reward: {e}")
        # TODO (Shankha): Decide if we should raise or return default score
        return 0.0


# Optional: Custom reward model for more sophisticated scoring
class OrchestratorRewardModel(nn.Module):
    """Neural reward model for orchestrator outputs."""
    
    def __init__(self, config):
        super().__init__()
        # Could implement a learned reward model here
        # For now, we use the simple compute_score function
        pass
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use compute_score function instead")
