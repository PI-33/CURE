# the code base is adapted from open-reasoner-zero (orz) and openrlhf

import os
import re
import sys
import ast
import ray
import json
import copy
import torch
import asyncio
import argparse
import numpy as np
from typing import List
from loguru import logger
from jinja2 import Template
from dataclasses import dataclass
from collections import defaultdict
from functools import cached_property
from typing_extensions import override
from itertools import islice, zip_longest
from omegaconf.listconfig import ListConfig
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, List, Optional, Tuple


from train_utils.dataset import PromptDataset
from train_utils.rl.trainer import RayPPOTrainer
from train_utils.base_exp import BasePPOExp, BasePPOExpConfig



import optimization_config
from runtime_paths import temp_data_dir, experiment_ckpt_root



# our optimization scripts are in ./train_utils/rl/...

@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True

    # Conditional settings with production values first
    total_num_nodes: int = optimization_config.total_num_nodes

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = optimization_config.actor_num_gpus_per_node
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = optimization_config.actor_num_gpus_per_node
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = False
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = optimization_config.pretrained_model

    rl_data: Optional[str] = "temp_data/rl_data.json"
    separate_training: bool = optimization_config.separate_training
    rl_code_data: Optional[str] = "temp_data/rl_code_data.json"
    rl_case_data: Optional[str] = "temp_data/rl_case_data.json"

    ckpt_path: str = "ckpt"
    save_path: str = "ckpt"
    tensorboard_log_dir: str = "tb_logs"

    # ppo related settings
    actor_learning_rate: float = optimization_config.actor_learning_rate
    num_warmup_steps: int = optimization_config.num_warmup_steps
    prompt_max_len: int = optimization_config.prompt_max_len
    enable_prefix_caching: bool = True
    advantage_normalize: bool = False

    policy_update_steps: int = optimization_config.policy_update_steps
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = optimization_config.kl_loss_coef
    use_kl_loss: bool = optimization_config.use_kl_loss
    use_kl_estimator_k3: bool = optimization_config.use_kl_estimator_k3

    # generate related settings
    generate_max_len: int = optimization_config.generate_max_len
    packing_max_len: int = optimization_config.packing_max_len
    
    max_epochs: int = optimization_config.max_epochs # number of iteration of training

    gamma: float = 1.0
    lambd: float = 1.0

    optimized_model_name: str = optimization_config.optimized_model_name



class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        return RayPPOTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            colocate_pg=self.get_colocate_pg,
        )


if __name__ == "__main__":

    import argparse, sys, asyncio, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tb_dir", type=str, default=None)
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown

    cfg = PPOExpConfig()
    if args.pretrain is not None:
        cfg.pretrain = args.pretrain
    cfg.global_step = args.step
    if args.tb_dir is not None:
        cfg.tensorboard_log_dir = args.tb_dir

    td = temp_data_dir()
    cfg.rl_data = os.path.join(td, "rl_data.json")
    cfg.rl_code_data = os.path.join(td, "rl_code_data.json")
    cfg.rl_case_data = os.path.join(td, "rl_case_data.json")
    # actors.save_model 会写入 join(save_path, optimized_model_name)，故此处必须是 ckpt 父目录，不能已是 .../ckpt/optimized
    ckpt_parent = experiment_ckpt_root()
    cfg.ckpt_path = ckpt_parent
    cfg.save_path = ckpt_parent

    exp = PPOExp().set_cfg(cfg)
    
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())
