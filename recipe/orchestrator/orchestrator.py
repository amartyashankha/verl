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
# from verl.utils.reward_score.grader import Grader  # Not available in current VERL version

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class OrchestratorDataset(RLHFDataset):
    """Dataset for orchestrator notebook agent that only needs an instance_id.

    Note: We accept a tokenizer argument for constructor compatibility with the
    base loader, but we do not use it. This dataset does not perform tokenization.
    """

    def __init__(self, data_files, tokenizer, config, processor=None, **kwargs):
        """Initialize orchestrator dataset.

        Args:
            data_files: Path(s) to data file(s)
            tokenizer: Unused. Present for compatibility.
            config: Configuration object containing dataset settings
            processor: Unused for this dataset
        """
        # Extract orchestrator-specific config from the main config
        self.enable_truncation = getattr(config, 'enable_truncation', False)
        self.truncation_strategy = getattr(config, 'truncation_strategy', "ast_llm_compaction")
        self.truncation_max_tokens = getattr(config, 'truncation_max_tokens', 16000)
        self.modal_base_url = getattr(config, 'modal_base_url', None)
        self.modal_evaluation_url = getattr(config, 'modal_evaluation_url', None)

        # Call parent init to leverage file downloading paths, but tokenization will be skipped
        super().__init__(data_files, tokenizer, config, processor, **kwargs)

    def _read_files_and_tokenize(self):
        """Read JSON files that may be a list-of-strings and normalize to minimal records.

        We keep only non-tensor fields and do not tokenize. Each row must yield an
        `instance_id` string. Other fields are optional and can be used by the agent loop.
        """
        dataframes = []
        for data_file in self.data_files:
            dataframe = datasets.load_dataset("json", data_files=data_file)["train"]
            # Normalize rows into a minimal schema; robust to list-of-strings inputs
            dataframe = dataframe.map(self.map_fn, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)
        logger.info(f"Loaded {len(self.dataframe)} instances for orchestrator (instance_id-only)")

    def map_fn(self, row: dict):
        """Normalize a dataset row to a minimal schema with just an instance_id.

        Supports JSON that is a list of strings. HF will expose such lists under
        a single column (commonly 'text'). We try common keys and also handle the
        single-column case gracefully.
        """
        instance_id = None
        if isinstance(row, dict):
            # Try common keys first
            for key in ("instance_id", "text", "id", "name"):
                val = row.get(key)
                if isinstance(val, str) and val:
                    instance_id = val
                    break
            # If there's only one column, take its value as instance_id
            if instance_id is None and len(row) == 1:
                instance_id = str(next(iter(row.values())))
        else:
            # Fallback if a bare value is encountered
            instance_id = str(row)

        if not instance_id:
            raise ValueError(f"Could not infer instance_id from row: {row}")

        return {
            "data_source": "swegym",
            "ability": "CODING",
            "agent_name": "orchestrator_coding_agent",
            "instance_id": instance_id,
            "run_id": f"run_{instance_id}",
            "notebook_id": "main",
            # Defaults for evaluation context; agent/reward loop can override/ignore
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            # Pass-through agent config if needed downstream
            "agent_config": {
                "enable_truncation": self.enable_truncation,
                "truncation_strategy": self.truncation_strategy,
                "truncation_max_tokens": self.truncation_max_tokens,
            },
            "modal_base_url": self.modal_base_url,
            "modal_evaluation_url": self.modal_evaluation_url,
        }

    def __getitem__(self, idx):
        """Return a minimal non-tensor sample with only orchestration metadata.

        We avoid the base class tokenization path entirely.
        """
        row_dict: dict = self.dataframe[idx]
        # Ensure required field exists
        instance_id = row_dict.get("instance_id")
        if not instance_id:
            raise ValueError(f"instance_id missing at index {idx}")
        return row_dict


async def compute_score(data_dict: dict, response: Union[str, list[dict]], grader=None, **kwargs) -> float:
    """Compute reward score for orchestrator agent responses.
    
    This is called after the agent completes its notebook execution.
    The solution patch is extracted by the agent loop and passed through the pipeline.
    
    Scoring strategy:
    - Resolved instance: 1.0
    - Not resolved but some tests pass: Up to 0.5 (based on % of tests passed)
    - No patch or evaluation failure: 0.0
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
    
    # TOREVIEW (Jeffrey): Get Modal evaluation URL from config
    modal_evaluation_url = data_dict.get("modal_evaluation_url", "https://fairies--evaluation-service-evaluate-patch.modal.run")
    
    try:
        # Call Modal evaluation endpoint for SWE-bench instances
        async with httpx.AsyncClient(timeout=600) as client:  # Increased timeout for evaluation
            reward_response = await client.post(
                modal_evaluation_url,
                json={
                    "instance_id": data_dict.get("instance_id"),
                    "patch": solution_patch,
                    "dataset_name": data_dict.get("dataset_name", "princeton-nlp/SWE-bench_Verified"),
                    "split": data_dict.get("split", "test"),
                    # Run ID can be passed for tracking
                    "run_id": data_dict.get("run_id", f"verl_eval_{data_dict.get('instance_id', 'unknown')}")
                }
            )
            
            if reward_response.status_code == 200:
                reward_data = reward_response.json()
                
                # Check if evaluation was successful
                if not reward_data.get("success", False):
                    logger.error(f"Evaluation failed: {reward_data.get('error', 'Unknown error')}")
                    return 0.0
                
                # Extract evaluation results
                resolved = reward_data.get("resolved", False)
                test_results = reward_data.get("test_results", {})
                execution_time = reward_data.get("execution_time", 0)
                
                # Log evaluation details
                logger.info(f"Evaluation completed in {execution_time:.2f}s")
                logger.info(f"Instance resolved: {resolved}")
                logger.info(f"Test results: {test_results}")
                
                # Convert resolution status to score
                # TODO (Shankha): Consider more nuanced scoring based on test results
                # For now: resolved = 1.0, not resolved = 0.0
                score = 1.0 if resolved else 0.0
                
                # Optional: Partial credit based on test results
                if not resolved and test_results:
                    # Count passed tests for partial credit
                    total_tests = len(test_results)
                    passed_tests = sum(1 for result in test_results.values() if result == "PASSED")
                    if total_tests > 0:
                        partial_score = passed_tests / total_tests * 0.5  # Max 0.5 for partial success
                        score = max(score, partial_score)
                        logger.info(f"Partial credit: {passed_tests}/{total_tests} tests passed = {partial_score}")
                
                return score
            else:
                logger.error(f"Evaluation request failed with status {reward_response.status_code}")
                logger.error(f"Response: {reward_response.text}")
                return 0.0
                
    except httpx.TimeoutException:
        logger.error("Evaluation timed out after 600 seconds")
        # TODO (Shankha): Implement retry logic or fallback scoring
        # Evaluation can take several minutes for complex patches
        return 0.0
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
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
