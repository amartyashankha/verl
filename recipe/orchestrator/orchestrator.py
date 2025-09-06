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
    """Wrapper that handles both sync and async contexts for compute_score."""
    import asyncio
    import httpx
    
    # Synchronous implementation for when we're already in an async context
    def sync_evaluate(instance_id, solution_patch, dataset_name, split, run_id, modal_evaluation_url):
        logger.info(f"[SYNC_EVAL] Using synchronous evaluation for {instance_id}")
        try:
            with httpx.Client(timeout=600) as client:
                logger.info(f"[SYNC_EVAL] Submitting to {modal_evaluation_url}/evaluate")
                response = client.post(
                    f"{modal_evaluation_url}/evaluate",
                    json={
                        "instance_id": instance_id,
                        "patch": solution_patch,
                        "dataset_name": dataset_name,
                        "split": split,
                        "run_id": run_id
                    }
                )
                
                if response.status_code == 200:
                    reward_data = response.json()
                    if not reward_data.get("success", False):
                        logger.error(f"[SYNC_EVAL] Failed: {reward_data.get('error')}")
                        return 0.0
                    
                    # Extract results and calculate score
                    resolved = reward_data.get("resolved", False)
                    tests_passed = reward_data.get("tests_passed", {})
                    tests_failed = reward_data.get("tests_failed", {})
                    
                    fail_to_pass = tests_passed.get("fail_to_pass", [])
                    pass_to_fail = tests_failed.get("pass_to_fail", [])
                    fail_to_fail = tests_failed.get("fail_to_fail", [])
                    
                    total_originally_failing = len(fail_to_pass) + len(fail_to_fail)
                    
                    logger.info(f"[SYNC_EVAL] Resolved: {resolved}, F2P: {len(fail_to_pass)}, P2F: {len(pass_to_fail)}")
                    
                    if resolved:
                        return 1.0
                    elif total_originally_failing > 0:
                        return len(fail_to_pass) / total_originally_failing
                    else:
                        return 0.0
                else:
                    logger.error(f"[SYNC_EVAL] HTTP {response.status_code}: {response.text[:200]}")
                    return 0.0
        except Exception as e:
            logger.error(f"[SYNC_EVAL] Error: {e}")
            return 0.0
    
    # Extract info from extra_info
    instance_id = "unknown"
    metrics = {}
    dataset_name = "SWE-Gym/SWE-Gym"
    split = "train"
    run_id = None
    modal_evaluation_url = None
    
    if extra_info:
        if hasattr(extra_info, '__len__') and not isinstance(extra_info, dict):
            extra_info = extra_info[0] if len(extra_info) > 0 else {}
        
        instance_id = extra_info.get("instance_id", "unknown")
        metrics = extra_info.get("metrics", {})
        dataset_name = extra_info.get("dataset_name", "SWE-Gym/SWE-Gym")
        split = extra_info.get("split", "train")
        run_id = extra_info.get("run_id", f"verl_eval_{instance_id}")
        modal_evaluation_url = extra_info.get("modal_evaluation_url")
    
    solution_patch = metrics.get("solution_patch", "")
    
    # Check for skipped instances
    if metrics.get("skipped"):
        logger.info(f"[COMPUTE_SCORE] Skipped instance {instance_id}: {metrics.get('reason')}")
        return 0.0
    
    if not solution_patch:
        logger.warning(f"[COMPUTE_SCORE] No solution patch for {instance_id}")
        return 0.0
    
    if not modal_evaluation_url:
        modal_evaluation_url = "https://fairies--swe-gym-evaluation-service-polling-fastapi-app.modal.run"
    
    logger.info(f"[COMPUTE_SCORE] Evaluating {instance_id} with patch length {len(solution_patch)}")
    
    # Always use synchronous evaluation since we're called from sync context
    return sync_evaluate(instance_id, solution_patch, dataset_name, split, run_id, modal_evaluation_url)

# Old async implementation removed - using simpler sync approach in compute_score above
    
    # Extract necessary information from extra_info
    instance_id = "unknown"
    metrics = {}
    dataset_name = "SWE-Gym/SWE-Gym"
    split = "train"
    run_id = None
    modal_evaluation_url = None
    
    if extra_info:
        # Debug: Log extra_info structure
        logger.debug(f"[DEBUG] extra_info structure:")
        logger.debug(f"  Type: {type(extra_info)}")
        logger.debug(f"  Is dict: {isinstance(extra_info, dict)}")
        logger.debug(f"  Has __len__: {hasattr(extra_info, '__len__')}")
        
        # Handle both dict and array-wrapped dict
        if hasattr(extra_info, '__len__') and not isinstance(extra_info, dict):
            logger.debug(f"  Array-wrapped, length: {len(extra_info)}")
            extra_info = extra_info[0] if len(extra_info) > 0 else {}
            logger.debug(f"  Unwrapped to type: {type(extra_info)}")
        
        # Debug: Log extracted values
        logger.debug(f"[DEBUG] extra_info keys: {list(extra_info.keys()) if isinstance(extra_info, dict) else 'Not a dict'}")
        
        instance_id = extra_info.get("instance_id", "unknown")
        metrics = extra_info.get("metrics", {})
        dataset_name = extra_info.get("dataset_name", "SWE-Gym/SWE-Gym")
        split = extra_info.get("split", "train")
        run_id = extra_info.get("run_id", f"verl_eval_{instance_id}")
        
        # Try to get modal_evaluation_url from extra_info
        modal_evaluation_url = extra_info.get("modal_evaluation_url")
        
        logger.debug(f"[DEBUG] Extracted from extra_info:")
        logger.debug(f"  instance_id: {instance_id}")
        logger.debug(f"  dataset_name: {dataset_name}")
        logger.debug(f"  split: {split}")
        logger.debug(f"  run_id: {run_id}")
        logger.debug(f"  modal_evaluation_url: {modal_evaluation_url}")
        logger.debug(f"  metrics keys: {list(metrics.keys()) if isinstance(metrics, dict) else 'Not a dict'}")
    
    # Extract solution patch from metrics
    solution_patch = metrics.get("solution_patch", "")
    solution_metadata = metrics.get("solution_metadata", {})
    
    logger.info(f"[COMPUTE_SCORE] Computing reward for instance {instance_id}")
    logger.debug(f"[DEBUG] Solution patch extraction:")
    logger.debug(f"  Patch length: {len(solution_patch)} chars")
    logger.debug(f"  Patch preview: {solution_patch[:200]}..." if solution_patch else "  No patch")
    logger.debug(f"  Solution metadata: {solution_metadata}")
    
    if not solution_patch:
        logger.warning(f"[COMPUTE_SCORE] No solution patch found for {instance_id}, returning 0.0")
        logger.debug(f"[DEBUG] Metrics contained: {list(metrics.keys()) if metrics else 'No metrics'}")
        return 0.0
    
    # Use default Modal evaluation URL if not provided
    if not modal_evaluation_url:
        modal_evaluation_url = "https://fairies--swe-gym-evaluation-service-polling-fastapi-app.modal.run"
        logger.debug(f"[DEBUG] Using default Modal evaluation URL: {modal_evaluation_url}")
    else:
        logger.debug(f"[DEBUG] Using provided Modal evaluation URL: {modal_evaluation_url}")
    
    # Define async function for evaluation
    async def evaluate_patch():
        try:
            # Call Modal evaluation endpoint for SWE-bench/SWE-Gym instances
            async with httpx.AsyncClient(timeout=600) as client:  # 10 minute timeout for evaluation
                logger.info(f"[EVALUATION] Submitting patch for evaluation to {modal_evaluation_url}")
                
                # Debug: Log request payload
                request_payload = {
                    "instance_id": instance_id,
                    "patch": solution_patch,
                    "dataset_name": dataset_name,
                    "split": split,
                    "run_id": run_id
                }
                logger.debug(f"[DEBUG] Request payload:")
                logger.debug(f"  instance_id: {instance_id}")
                logger.debug(f"  dataset_name: {dataset_name}")
                logger.debug(f"  split: {split}")
                logger.debug(f"  run_id: {run_id}")
                logger.debug(f"  patch length: {len(solution_patch)} chars")
                
                reward_response = await client.post(
                    f"{modal_evaluation_url}/evaluate",
                    json=request_payload
                )
                
                logger.debug(f"[DEBUG] Response status code: {reward_response.status_code}")
                
                if reward_response.status_code == 200:
                    reward_data = reward_response.json()
                    logger.debug(f"[DEBUG] Response data keys: {list(reward_data.keys())}")
                    
                    # Check if evaluation was successful
                    if not reward_data.get("success", False):
                        logger.error(f"[EVALUATION] Failed: {reward_data.get('error', 'Unknown error')}")
                        logger.debug(f"[DEBUG] Full error response: {reward_data}")
                        return 0.0
                    
                    # Extract evaluation results
                    resolved = reward_data.get("resolved", False)
                    test_results = reward_data.get("test_results", {})
                    execution_time = reward_data.get("execution_time", 0)
                    
                    # Extract fail-to-pass metrics (standard SWE-bench evaluation)
                    tests_passed = reward_data.get("tests_passed", {})
                    tests_failed = reward_data.get("tests_failed", {})
                    
                    # Calculate fail-to-pass ratio
                    # These are tests that were originally failing but now pass with the patch
                    fail_to_pass = tests_passed.get("fail_to_pass", [])
                    pass_to_pass = tests_passed.get("pass_to_pass", [])
                    
                    # These are tests that should not fail (originally passing tests)
                    pass_to_fail = tests_failed.get("pass_to_fail", [])
                    fail_to_fail = tests_failed.get("fail_to_fail", [])
                    
                    # Total originally failing tests
                    total_originally_failing = len(fail_to_pass) + len(fail_to_fail)
                    
                    # Log evaluation details
                    logger.info(f"[EVALUATION] Completed in {execution_time:.2f}s")
                    logger.info(f"[EVALUATION] Instance resolved: {resolved}")
                    logger.info(f"[EVALUATION] Test breakdown:")
                    logger.info(f"  Fail-to-pass: {len(fail_to_pass)} tests (fixed)")
                    logger.info(f"  Pass-to-pass: {len(pass_to_pass)} tests (still passing)")
                    logger.info(f"  Pass-to-fail: {len(pass_to_fail)} tests (regressions)")
                    logger.info(f"  Fail-to-fail: {len(fail_to_fail)} tests (still failing)")
                    
                    # Debug: Log actual test names if available
                    if fail_to_pass:
                        logger.debug(f"[DEBUG] Fail-to-pass tests: {fail_to_pass[:5]}{'...' if len(fail_to_pass) > 5 else ''}")
                    if pass_to_fail:
                        logger.debug(f"[DEBUG] Pass-to-fail tests (regressions): {pass_to_fail[:5]}{'...' if len(pass_to_fail) > 5 else ''}")
                    
                    # Calculate reward based on fail-to-pass ratio
                    # Standard SWE-bench scoring:
                    # - 1.0 if resolved (all originally failing tests now pass, no regressions)
                    # - Otherwise, use fail-to-pass ratio with penalties for regressions
                    
                    if resolved:
                        score = 1.0
                        logger.info(f"[SCORING] Instance fully resolved! Score: {score}")
                    else:
                        # Calculate base score from fail-to-pass ratio
                        if total_originally_failing > 0:
                            fail_to_pass_ratio = len(fail_to_pass) / total_originally_failing
                            base_score = fail_to_pass_ratio
                            
                            logger.debug(f"[DEBUG] Scoring calculation:")
                            logger.debug(f"  Total originally failing: {total_originally_failing}")
                            logger.debug(f"  Fail-to-pass ratio: {fail_to_pass_ratio:.3f}")
                            logger.debug(f"  Base score: {base_score:.3f}")
                            
                            # Apply penalty for regressions (pass-to-fail tests)
                            # Each regression reduces the score
                            if len(pass_to_fail) > 0:
                                total_originally_passing = len(pass_to_pass) + len(pass_to_fail)
                                if total_originally_passing > 0:
                                    regression_penalty = len(pass_to_fail) / total_originally_passing * 0.5
                                    base_score = max(0, base_score - regression_penalty)
                                    logger.info(f"[SCORING] Applied regression penalty: -{regression_penalty:.3f} for {len(pass_to_fail)} regressions")
                                    logger.debug(f"[DEBUG] Total originally passing: {total_originally_passing}")
                                    logger.debug(f"[DEBUG] Score after penalty: {base_score:.3f}")
                            
                            score = base_score
                            logger.info(f"[SCORING] Fail-to-pass ratio: {fail_to_pass_ratio:.3f} ({len(fail_to_pass)}/{total_originally_failing})")
                            logger.info(f"[SCORING] Final score: {score:.3f}")
                        else:
                            # No originally failing tests (shouldn't happen in SWE-bench)
                            # Check if there are any regressions
                            if len(pass_to_fail) > 0:
                                score = 0.0
                                logger.warning(f"No originally failing tests, but {len(pass_to_fail)} regressions found")
                            else:
                                score = 0.5  # Neutral score if no tests to fix and no regressions
                                logger.warning("No originally failing tests to evaluate")
                    
                    # Fallback to old logic if new fields are not present
                    if not tests_passed and not tests_failed and test_results:
                        logger.info("Using fallback scoring based on test_results")
                        # Count passed tests for partial credit
                        total_tests = len(test_results)
                        passed_tests = sum(1 for result in test_results.values() if result == "PASSED")
                        if total_tests > 0:
                            partial_score = passed_tests / total_tests * 0.5  # Max 0.5 for partial success
                            score = max(score, partial_score)
                            logger.info(f"Fallback partial credit: {passed_tests}/{total_tests} tests passed = {partial_score}")
                    
                    return score
                else:
                    logger.error(f"[EVALUATION] Request failed with status {reward_response.status_code}")
                    logger.error(f"[EVALUATION] Response: {reward_response.text[:500]}...")
                    logger.debug(f"[DEBUG] Full response: {reward_response.text}")
                    return 0.0
                    
        except httpx.TimeoutException as e:
            logger.error(f"[EVALUATION] Timed out after 600 seconds: {e}")
            logger.debug(f"[DEBUG] Timeout details: {str(e)}")
            return 0.0
        except Exception as e:
            logger.error(f"[EVALUATION] Error during evaluation: {e}")
            logger.debug(f"[DEBUG] Exception type: {type(e).__name__}")
            logger.debug(f"[DEBUG] Exception details: {str(e)}")
            import traceback
            logger.debug(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            return 0.0
    
    # Run the async evaluation
    try:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            logger.debug("[DEBUG] Already in event loop, using nest_asyncio to allow nested async execution")
            # Use nest_asyncio to allow nested event loops
            import nest_asyncio
            nest_asyncio.apply()
            # Now we can run the async function even in an existing loop
            score = asyncio.run(evaluate_patch())
            logger.info(f"[COMPUTE_SCORE] Returning final score: {score:.3f} for {instance_id}")
            return score
        except RuntimeError:
            # No event loop running, we can use asyncio.run()
            logger.debug("[DEBUG] No event loop running, using asyncio.run()")
            score = asyncio.run(evaluate_patch())
            logger.info(f"[COMPUTE_SCORE] Returning final score: {score:.3f} for {instance_id}")
            return score
    except ImportError as e:
        logger.error("[COMPUTE_SCORE] nest_asyncio not installed. Install with: pip install nest-asyncio")
        logger.error("[COMPUTE_SCORE] Falling back to synchronous evaluation with httpx")
        # Fallback to synchronous evaluation
        import httpx
        try:
            with httpx.Client(timeout=600) as client:
                logger.info(f"[EVALUATION] Submitting patch for SYNC evaluation to {modal_evaluation_url}")
                request_payload = {
                    "instance_id": instance_id,
                    "patch": solution_patch,
                    "dataset_name": dataset_name,
                    "split": split,
                    "run_id": run_id
                }
                response = client.post(f"{modal_evaluation_url}/evaluate", json=request_payload)
                if response.status_code == 200:
                    reward_data = response.json()
                    if not reward_data.get("success", False):
                        logger.error(f"[EVALUATION] Failed: {reward_data.get('error', 'Unknown error')}")
                        return 0.0
                    
                    # Use simplified scoring for sync fallback
                    resolved = reward_data.get("resolved", False)
                    score = 1.0 if resolved else 0.0
                    logger.info(f"[EVALUATION] Sync evaluation completed. Resolved: {resolved}, Score: {score}")
                    return score
                else:
                    logger.error(f"[EVALUATION] Sync request failed with status {response.status_code}")
                    return 0.0
        except Exception as sync_e:
            logger.error(f"[EVALUATION] Sync evaluation also failed: {sync_e}")
            return 0.0
    except Exception as e:
        logger.error(f"[COMPUTE_SCORE] Failed to run evaluation: {e}")
        logger.debug(f"[DEBUG] Exception during asyncio execution: {type(e).__name__}: {str(e)}")
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

