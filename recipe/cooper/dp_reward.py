
"""
Implement a multiprocess PPOCritic
"""

import itertools
import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# We borrow code from dp_critic.py
class DataParallelPPOReward():
    def __init__(self, config, reward_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        assert self.use_remove_padding == False
        print(f"Reward use_remove_padding={self.use_remove_padding}")

        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch):
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
                        
            if self.use_remove_padding:
                raise NotImplementedError
            else:
                output = self.reward_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )  # prevent model thinks we are generating
            return output.logits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.reward_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.reward_optimizer.zero_grad()
        else:
            self.reward_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp reward", logger=logger)
    def compute_rewards(self, data: DataProto) -> torch.Tensor: # (bs,)
        self.reward_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)

        reward_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                score = self._forward_micro_batch(micro_batch)
            reward_lst.append(score)
        rewards = torch.concat(reward_lst, dim=0).squeeze()
        rewards = torch.sigmoid(rewards)  # (bs,)

        if self.config.reward_scale:  # scale the reward to 0 or 1
            rewards = (rewards > 0.5).float()
            
        return rewards

    @GPUMemoryLogger(role="dp reward", logger=logger)
    def update_reward(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        select_keys = [
            "positive_input_ids", "positive_attention_mask", "positive_position_ids",
            "negative_input_ids", "negative_attention_mask", "negative_position_ids",
            "valid_mask"
        ]
        
        batch = data.select(batch_keys=select_keys).batch
        print(len(batch), self.config.ppo_mini_batch_size)
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        
        pbar = tqdm(desc='Update reward', total=len(dataloader) * self.config.ppo_epochs)
        
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                self.reward_optimizer.zero_grad()

                for mini_batch in batches:
                    # Support all devices
                    pos = {
                        "input_ids": mini_batch["positive_input_ids"].to(get_device_id()),
                        "attention_mask": mini_batch["positive_attention_mask"].to(get_device_id()),
                        "position_ids": mini_batch["positive_position_ids"].to(get_device_id()),
                    }
                    neg = {
                        "input_ids": mini_batch["negative_input_ids"].to(get_device_id()),
                        "attention_mask": mini_batch["negative_attention_mask"].to(get_device_id()),
                        "position_ids": mini_batch["negative_position_ids"].to(get_device_id()),
                    }
                    eps = 1e-6  # to avoid division by zero
                    positive_scores = self._forward_micro_batch(pos) # (bs,)
                    negative_scores = self._forward_micro_batch(neg) # (bs,)
                    score_diff = positive_scores - negative_scores  # (batch_size,)
                    valid_mask = mini_batch["valid_mask"].to(get_device_id()).float()
                    loss = (-torch.nn.functional.logsigmoid(score_diff) * valid_mask).sum()/(valid_mask.sum()+eps)     
                    loss = loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "reward/loss": loss.detach().item(),
                    }

                    append_to_dict(metrics, data)
                    
                pbar.update(1)
                grad_norm = self._optimizer_step()
                data = {"reward/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        pbar.close()
        self.reward_optimizer.zero_grad()
        return metrics
