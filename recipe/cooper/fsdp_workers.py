
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
from typing import Union
from openai import OpenAI
import numpy as np

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.workers.fsdp_workers import RewardModelWorker as verl_RewardModelWorker
from verl.utils.device import get_device_id
from .dp_reward import DataParallelPPOReward

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def cooper_format_prompt(question, answer, completion):
    if len(answer) > 1500:
        answer = answer[:500] + answer[-1000:]
    if len(completion) > 1500:
        completion = completion[:500] + completion[-1000:]
        
    return f"<question>{question}</question>\n<reference_answer>{answer}</reference_answer>\n<completion>{completion}</completion>"


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(verl_RewardModelWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__(config)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        
        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"

    def _build_reward_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim

        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForSequenceClassification
        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType
        
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
                num_labels=1,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)
            
            if config.model.get("enable_gradient_checkpointing", False):
                reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
        if self.rank == 0:
            print_model_size(reward_module)
        
        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        log_gpu_memory_usage("Before Reward FSDP", logger=None)
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        print(reward_module.device)
        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(reward_module, config.strategy, enable_gradient_checkpointing)
        
        log_gpu_memory_usage("After reward FSDP", logger=None)
        
        reward_optimizer = optim.AdamW(
            reward_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
        if warmup_style == "constant":
            reward_lr_scheduler = get_constant_schedule_with_warmup(optimizer=reward_optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":  # cosine decay
            reward_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=reward_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        
        return reward_module, reward_optimizer, reward_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module, self.reward_optimizer, self.reward_lr_scheduler = self._build_reward_model_optimizer(config=self.config)
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)
        
        self.reward = DataParallelPPOReward(
            config=self.config, 
            reward_module=self.reward_module, 
            reward_optimizer=self.reward_optimizer)
        
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.reward_module,
            optimizer=self.reward_optimizer,
            lr_scheduler=self.reward_lr_scheduler,
            processing_class=self.input_tokenizer if self._do_switch_chat_template else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    #update_Reward
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_reward(self, data: DataProto):
        """
        Update the reward model with the given data.
        Args：
            data (DataProto): The data to update the reward model with.
                positive_input_ids, positive_attention_mask, positive_position_ids
                negative_input_ids, negative_attention_mask, negative_position_ids     
        """
        valid_rate = sum(data.non_tensor_batch["is_valid"])/ len(data.non_tensor_batch["is_valid"])
        valid_mask = data.batch["valid_mask"]
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)  
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.reward_optimizer, device_id=get_device_id())
        
        if self._do_switch_chat_template: # switch chat template before process
            positive_data = DataProto.from_dict({
                "input_ids": data.batch["positive_input_ids"],
                "attention_mask": data.batch["positive_attention_mask"],
                "position_ids": data.batch["positive_position_ids"],
                "responses":  data.batch["positive_responses"],
            }, non_tensors=data.non_tensor_batch)
            negative_data = DataProto.from_dict({
                "input_ids": data.batch["negative_input_ids"],
                "attention_mask": data.batch["negative_attention_mask"],
                "position_ids": data.batch["negative_position_ids"],
                "responses":  data.batch["negative_responses"]
            }, non_tensors=data.non_tensor_batch)
            
            positive_data = self._switch_chat_template(positive_data)
            negative_data = self._switch_chat_template(negative_data)
            data = DataProto.from_dict({
                "positive_input_ids": positive_data.batch["input_ids"],
                "positive_attention_mask": positive_data.batch["attention_mask"],
                "positive_position_ids": positive_data.batch["position_ids"],
                
                "negative_input_ids": negative_data.batch["input_ids"],
                "negative_attention_mask": negative_data.batch["attention_mask"],
                "negative_position_ids": negative_data.batch["position_ids"],
            }).to(get_device_id())
            data.batch["valid_mask"] = valid_mask
        self.reward_optimizer.zero_grad()  
        
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_reward", logger=None) as timer:
                metrics = self.reward.update_reward(data=data)

            self.reward_lr_scheduler.step()
            lr = self.reward_lr_scheduler.get_last_lr()[0]
            metrics["reward/lr"] = lr
            metrics["reward/valid_rate"] = valid_rate

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        output = output.to("cpu")
        return output
            
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
        
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        rm_data.meta_info["micro_batch_size"] = self.config.forward_micro_batch_size_per_gpu
        rm_data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        rm_data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        
        # Support all hardwares
        rm_data.batch = rm_data.batch.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            
            scores = self.reward.compute_rewards(data=rm_data)
            
            token_level_scores = self._expand_to_token_level(data, scores)
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        return output
            
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
            
        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)

    def _switch_chat_template(self, data: DataProto):
        if not self.config.use_cooper_template:
            super()._switch_chat_template(data)

        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response = src_tokenizer.decode(valid_response_ids)
            response = response.replace(src_tokenizer.eos_token, "")
            
            question = chat[0]["content"]
            answer = data.non_tensor_batch["reward_model"][i]['ground_truth']

            prompt_with_chat_template = cooper_format_prompt(question, answer, response)
                
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)
