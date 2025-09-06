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

    Note: We accept a tokenWizer argument for constructor compatibility with the
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
        
        # Ensure we have a valid evaluation URL
        if not self.modal_evaluation_url:
            self.modal_evaluation_url = "https://fairies--swe-gym-evaluation-service-polling-fastapi-app.modal.run"
            logger.info(f"No modal_evaluation_url in config, using default: {self.modal_evaluation_url}")
        
        # Pre-load SWE-bench/SWE-Gym datasets for efficient problem statement lookup
        self._problem_statements_cache = {}
        self._preload_problem_statements()

        # Call parent init to leverage file downloading paths, but tokenization will be skipped
        super().__init__(data_files, tokenizer, config, processor, **kwargs)
    
    def _preload_problem_statements(self):
        """Pre-load problem statements from SWE-Gym dataset into a cache."""
        try:
            from datasets import load_dataset
            
            # Load SWE-Gym only
            try:
                swe_gym = load_dataset("SWE-Gym/SWE-Gym", split="train")
                logger.info(f"Loading SWE-Gym dataset with {len(swe_gym)} instances")
                
                stored_count = 0
                missing_count = 0
                
                for item in swe_gym:
                    instance_id = item.get("instance_id")
                    if instance_id:
                        # Extract problem statement - SWE-Gym uses "problem_statement" field
                        problem_statement = item.get("problem_statement")
                        if problem_statement and isinstance(problem_statement, str) and problem_statement.strip():
                            self._problem_statements_cache[instance_id] = problem_statement
                            stored_count += 1
                        else:
                            # Fallback to other possible fields if problem_statement is missing
                            found_fallback = False
                            for key in ("problem", "prompt", "task_prompt", "description"):
                                if key in item and isinstance(item[key], str) and item[key].strip():
                                    self._problem_statements_cache[instance_id] = item[key]
                                    logger.debug(f"Used fallback field '{key}' for instance {instance_id}")
                                    stored_count += 1
                                    found_fallback = True
                                    break
                            if not found_fallback:
                                missing_count += 1
                                logger.debug(f"No problem statement found for instance {instance_id}")
                
                # Log summary and sample of what's in the cache
                logger.info(f"Pre-loaded {len(self._problem_statements_cache)} problem statements from SWE-Gym")
                logger.info(f"  Stored: {stored_count}, Missing: {missing_count}")
                
                # Show a few sample entries to verify cache contents
                if self._problem_statements_cache:
                    sample_items = list(self._problem_statements_cache.items())[:3]
                    for instance_id, problem in sample_items:
                        logger.info(f"  Sample cache entry: {instance_id} -> {problem[:100]}...")
                        
            except Exception as e:
                logger.warning(f"Failed to load SWE-Gym dataset: {e}")
                
        except ImportError:
            logger.warning("datasets library not available, problem statements will not be loaded")

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

    def map_fn(self, row):
        """Normalize a dataset row to a minimal schema with just an instance_id.

        Supports JSON that is a list of strings. HF will expose such lists under
        a single column (commonly 'text'). We try common keys and also handle the
        single-column case gracefully.
        """
        instance_id = None
        
        # Handle LazyRow objects from datasets library (they act like dicts but aren't dict instances)
        # Also handle regular dicts
        if hasattr(row, '__getitem__') and hasattr(row, 'keys'):
            # Try common keys first
            for key in ("instance_id", "text", "id", "name"):
                if key in row:
                    val = row[key]
                    if isinstance(val, str) and val:
                        instance_id = val
                        break
            
            # If there's only one column, take its value as instance_id
            if instance_id is None and len(row) == 1:
                # Get the first (and only) value
                val = row[next(iter(row.keys()))]
                if isinstance(val, str):
                    instance_id = val
                elif isinstance(val, (int, float)):
                    instance_id = str(val)
                else:
                    # This shouldn't happen, but log it for debugging
                    logger.warning(f"Unexpected value type in single-column row: {type(val).__name__} = {val!r}")
                    logger.warning(f"Row keys: {list(row.keys())}, Row: {dict(row)}")
                    raise ValueError(f"Could not extract instance_id from single-column row with value type {type(val).__name__}")
        elif isinstance(row, str):
            # If row is already a string, use it directly
            instance_id = row
        else:
            # This shouldn't happen - log the unexpected type
            logger.warning(f"Unexpected row type: {type(row).__name__}")
            logger.warning(f"Row has __getitem__: {hasattr(row, '__getitem__')}, has keys: {hasattr(row, 'keys')}")
            try:
                logger.warning(f"Row as dict: {dict(row)}")
            except:
                logger.warning(f"Could not convert row to dict")
            raise ValueError(f"Unexpected row type: {type(row).__name__}")

        if not instance_id:
            raise ValueError(f"Could not infer instance_id from row: {row!r}")
        
        # Determine dataset based on instance_id pattern
        # SWE-Gym instances typically don't have double underscores
        # SWE-bench instances have format: repo__issue
        dataset_name = "SWE-Gym/SWE-Gym" 
        split = "train" 
        
        # Get problem statement from cache
        # Debug: log what we're looking up
        logger.debug(f"Looking up instance_id: {instance_id!r} (type: {type(instance_id).__name__})")
        task_prompt = self._problem_statements_cache.get(instance_id)
        if not task_prompt:
            logger.debug(f"No problem statement found in cache for {instance_id!r}")
            # Also log a sample of what's in the cache for debugging
            if self._problem_statements_cache:
                sample_keys = list(self._problem_statements_cache.keys())[:3]
                logger.debug(f"Sample cache keys: {sample_keys}")

        return {
            "data_source": "swegym",
            "ability": "CODING",
            "agent_name": "orchestrator_coding_agent",
            "instance_id": instance_id,
            "run_id": f"run_{instance_id}",
            "notebook_id": "main",
            "dataset_name": dataset_name,
            "split": split,
            "task_prompt": task_prompt,  # Now included in the dataset output
            "modal_evaluation_url": self.modal_evaluation_url,  # Pass Modal evaluation URL
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


def compute_score(data_source: str, solution_str: str, ground_truth=None, extra_info=None, **kwargs) -> float:
    """Compute reward score for orchestrator agent responses.
    
    NOTE: The actual evaluation now happens directly in orchestrator_coding_agent_loop.py
    This function is only called if the agent loop didn't already compute the reward.
    
    Args:
        data_source: The data source identifier (e.g., "swegym")
        solution_str: The model's response (not used for orchestrator)
        ground_truth: Ground truth if available
        extra_info: Dictionary containing metrics and other info from agent loop
        **kwargs: Additional arguments
    
    Returns:
        float: Default reward of 0.0 (actual evaluation happens in agent loop)
    """
    # Log that we're in the fallback path
    instance_id = "unknown"
    if extra_info:
        if hasattr(extra_info, '__len__') and not isinstance(extra_info, dict):
            extra_info = extra_info[0] if len(extra_info) > 0 else {}
        instance_id = extra_info.get("instance_id", "unknown")
    
    logger.info(f"Fallback reward computation for {instance_id} - returning 0.0")
    logger.info("Note: Evaluation should have happened in orchestrator_coding_agent_loop.py")
    
    # Return default score since evaluation happens in the agent loop
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
