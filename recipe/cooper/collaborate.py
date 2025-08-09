from verl import DataProto
from openai import OpenAI,AsyncOpenAI
import asyncio
import time
import torch
import numpy as np
from tqdm import tqdm
from math_verify import parse, verify


class CollaborateDataGenerator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = config.model_name
        self.tokenizer = tokenizer
        self.concurrency_limit = config.concurrency_limit
        
    def _select_positive_response(self, data: DataProto):
        """
        Select the positive response from the data.
        Args:
            data (DataProto): The data to select the positive response from.
        Returns:
            DataProto: The data with the selected positive response.
        """
        # reward_tensor is token_level_reward (batch_size, seq_len)
        scores = data.batch["rule_based_scores"].sum(dim=-1)  # (batch_size,)
        batch_size = len(data)
        selected_data = {}
        uids = data.non_tensor_batch.get("uid", [f"uid_{i}" for i in range(batch_size)])

        print(f"Debug: Original batch size: {batch_size}")
        uids = data.non_tensor_batch.get("uid", [f"uid_{i}" for i in range(batch_size)])
        unique_uids = set(uids)
        print(f"Debug: Number of unique UIDs: {len(unique_uids)}")
        if len(unique_uids) > 0:
            print(f"Debug: Number of samples per UID: {batch_size / len(unique_uids):.2f}")

        uid_first_index = {}
        for i, uid in enumerate(uids):
            if uid not in uid_first_index:
                uid_first_index[uid] = i


        uid_positive_index = {}
        for i, uid in enumerate(uids):
            if scores[i].item() == 1:
                if uid not in uid_positive_index:
                    uid_positive_index[uid] = i


        positive_indices = []
        has_positive = []
        for i, uid in enumerate(unique_uids):
            if uid in uid_positive_index:
                positive_indices.append(uid_positive_index[uid])
                has_positive.append(True)
            else:
                positive_indices.append(uid_first_index[uid])
                has_positive.append(False)

        positive_indices_tensor = torch.tensor(positive_indices, device=data.batch.device)

        selected_tensor_batch = {}
        selected_non_tensor_batch = {}
        for key, value in data.batch.items():
            if isinstance(value, torch.Tensor):
                selected_tensor_batch[key] = value[positive_indices_tensor]
            else:
                selected_non_tensor_batch[key] = [value[i] for i in positive_indices_tensor.cpu().numpy()]
        for key, value in data.non_tensor_batch.items():
            if isinstance(value, (list, np.ndarray)):
                selected_non_tensor_batch[key] = [value[i] for i in positive_indices_tensor.cpu().numpy()]
            else:
                selected_non_tensor_batch[key] = value

        #  has_positive signifies whether the sample has a positive response
        selected_non_tensor_batch["has_positive"] = has_positive

        selected_dataproto = DataProto.from_dict(
            tensors=selected_tensor_batch,
            non_tensors=selected_non_tensor_batch,
            meta_info=data.meta_info.copy() if data.meta_info else {}
        )

        print(f"Debug: Successfully created DataProto with {len(selected_dataproto)} samples")
        return selected_dataproto


    def _generate_negative_response(self, data: DataProto):
        """
        Generate the negative response from the data.
        Args:
            data (DataProto): The data to generate the negative response from.
        Returns:
            data (DataProto): The data with the generated negative response.
        """
        # This function should be implemented to generate the negative response from the data.
        # For example, it can generate a random response or a response with lower reward score.
        
        # Extract batch information
        batch_size = data.batch.batch_size[0]
        negative_responses = []

        # Select positive responses maxlength 
        max_prompt_length = self.config.get("max_prompt_length", 512)
        max_response_length = self.config.get("max_response_length", 512)
        src_max_length = data.batch["attention_mask"].shape[-1]

        positive_data = []
        
        for i in range(batch_size):
            # Extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                messages: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                messages: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # Extract and decode positive response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length - 1] # we need to exclude the eos token
            positive_response = self.tokenizer.decode(valid_response_ids)
            
            assert isinstance(messages, list) and len(messages) == 1
            prompt_text = messages[0]['content']
            positive_data.append((prompt_text, positive_response))
        # Generate negative response using API
        start_time = time.time()
        negative_responses = asyncio.run(self._async_process_queries(positive_data))
        end_time = time.time()
        print(f"time for generate negative response: {end_time - start_time:.2f} seconds")


        is_valid = []
        has_positive = data.non_tensor_batch["has_positive"]
        for i in range(batch_size):
            if not has_positive[i] or negative_responses[i] is None:
                negative_responses[i] = positive_data[i][1]  # Use the positive response as fallback
                is_valid.append(False)
            else:
                is_valid.append(True)
        # Create a new DataProto with the negative responses
        new_batch = {}
        new_non_tensor_batch = {}

        # Process negative responses
        negative_input_ids = []
        negative_attention_mask = []
        negative_position_ids = []
        negative_response_ids = []
        
        raw_prompt = []

        for i, (negative_response, valid) in enumerate(zip(negative_responses, is_valid)):
            # Get the original chat for this sample
            if not valid:
                negative_input_ids.append(data.batch["input_ids"][i].unsqueeze(0))
                negative_attention_mask.append(data.batch["attention_mask"][i].unsqueeze(0))
                negative_position_ids.append(data.batch["position_ids"][i].unsqueeze(0))
                negative_response_ids.append(data.batch["responses"][i].unsqueeze(0))
                continue
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                messages = data.non_tensor_batch["raw_prompt"][i].copy()
            else:
                messages = data.non_tensor_batch["raw_prompt"][i].tolist()
            
            # Apply chat template
            prompt_with_negative = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            response_with_negative = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": negative_response}], 
                add_generation_prompt=False, tokenize=False
            )
            assert prompt_with_negative in response_with_negative
            response_with_negative = response_with_negative.replace(prompt_with_negative, "")
    
            prompt_inputs= self.tokenizer(
                prompt_with_negative, 
                return_tensors="pt", 
                add_special_tokens=False,
                max_length=max_prompt_length,
                truncation=True,
                padding_side="left",  # left padding for prompt
                padding="max_length"
            )
            prompt_input_ids = prompt_inputs["input_ids"]
            prompt_attention_mask = prompt_inputs["attention_mask"]

            response_with_negative_inputs = self.tokenizer(
                response_with_negative, 
                return_tensors="pt", 
                add_special_tokens=False,
                max_length=max_response_length,
                truncation=True,
                padding_side="right",  # right padding for response
                padding="max_length"
            )
            response_with_negative_input_ids = response_with_negative_inputs['input_ids']
            response_with_negative_attention_mask = response_with_negative_inputs['attention_mask']


            attention_mask = torch.cat([prompt_attention_mask, response_with_negative_attention_mask], dim=-1)  # (1, prompt_length + response_length)
            input_ids = torch.cat([prompt_input_ids, response_with_negative_input_ids], dim=-1)
            # Compute position ids
            position_ids = self.compute_position_id_from_first_valid_1d(attention_mask)  
            # (1, prompt_length + response_length):
            position_ids = position_ids.unsqueeze(0)


            negative_input_ids.append(input_ids)
            negative_attention_mask.append(attention_mask)
            negative_position_ids.append(position_ids)
            negative_response_ids.append(response_with_negative_input_ids)
            
            raw_prompt.append(messages)

        # Concatenate all negative samples
        negative_input_ids = torch.cat(negative_input_ids, dim=0)
        negative_attention_mask = torch.cat(negative_attention_mask, dim=0) 
        negative_position_ids = torch.cat(negative_position_ids, dim=0)
        negative_response_ids = torch.cat(negative_response_ids, dim=0)  # (batch_size, response_length)

        # Update batch with negative responses
        new_batch["input_ids"] = negative_input_ids
        new_batch["attention_mask"] = negative_attention_mask
        new_batch["position_ids"] = negative_position_ids
        new_batch["responses"] = negative_response_ids  # Use input_ids as responses for consistency
        new_batch["valid_mask"] = torch.tensor(is_valid).unsqueeze(-1).to(torch.int)
        
        for key, value in data.non_tensor_batch.items():
            new_non_tensor_batch[key] = value
        new_non_tensor_batch["is_valid"] = is_valid
        negative_data = DataProto.from_dict(
            tensors=new_batch,         
            non_tensors=new_non_tensor_batch,  
            meta_info=data.meta_info.copy()
        )
        return negative_data

    def preprocess_reward_data(self, data: DataProto) -> DataProto:
        # Implement your preprocessing logic here
        positive_data = self._select_positive_response(data)
        negative_data = self._generate_negative_response(positive_data)
        combined_tensors = {
            "positive_input_ids": positive_data.batch["input_ids"],
            "negative_input_ids": negative_data.batch["input_ids"],
            
            "positive_attention_mask": positive_data.batch["attention_mask"],
            "negative_attention_mask": negative_data.batch["attention_mask"],
            
            "positive_position_ids": positive_data.batch["position_ids"],
            "negative_position_ids": negative_data.batch["position_ids"],
            
            "positive_responses": positive_data.batch["responses"],
            "negative_responses": negative_data.batch["responses"],
            "valid_mask": negative_data.batch["valid_mask"],
        }

        combined_non_tensors = {}
        for key, value in positive_data.non_tensor_batch.items():
            combined_non_tensors[key] = value
        combined_non_tensors["is_valid"] = negative_data.non_tensor_batch["is_valid"]

        combined_data = DataProto.from_dict(
            tensors=combined_tensors,
            non_tensors=combined_non_tensors
        )
        return combined_data

    async def _call_api_for_negative_response(self, problem: str, positive_response: str, request_id: int, semaphore) -> tuple[int, str]:
        """
        Async call the API to get a negative response based on the prompt and positive response.
        Args:
            problem (str): The prompt to generate the negative response from.
            positive_response (str): The positive response to base the negative response on.
            request_id (int): The id of the request for tracking.
            semaphore: asyncio.Semaphore to limit concurrency.
        Returns:
            tuple[int, str]: The request id and the generated negative response.
        """
        async with semaphore:
            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    system_prompt = """You are tasked with generating a completely incorrect and misleading response. Given a math problem and a correct response, you must:
    1. Provide wrong reasoning and logic throughout
    2. Reach an incorrect conclusion or result
    3. Make the response similar in length to the correct response
    4. Ensure the response appears to solve the math problem but is factually wrong
    5. Use plausible-sounding but incorrect information
    """

                    user_message = f"""Math Problem: {problem}

    Correct Response: {positive_response}

    The following only needs to give the process of solving the question and the answer, do not give any irrelevant content."""

                    completion = await self.aclient.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=len(positive_response.split()) + 500,  # Match length approximately
                        temperature=0.7
                    )
                    negative_response = completion.choices[0].message.content
                    parsed_negative = parse(negative_response)
                    parsed_positive = parse(positive_response)
                    is_negative = not verify(parsed_negative, parsed_positive)
                    if not is_negative:
                        raise ValueError("Generated response is not negative enough.")
                    return request_id, negative_response
                except Exception as e:
                    retries += 1
                    print(f"ID {request_id} error: {e}, {retries} retrying...")
                    if retries >= max_retries:
                        print(f"ID {request_id} exceeded max retries, returning error.")
                        return request_id, None
                    await asyncio.sleep(0.01)  

    async def _async_process_queries(self, inputs: list[tuple[str, str]]) -> list[str]:
        """
        Asynchronous processing of queries with concurrency limit and progress tracking.
        Args:
            inputs: List of (problem, positive_response) tuples.
        Returns:
            List of generated negative responses (or None if failed).
        """
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # Create tasks for each input
        tasks = [asyncio.create_task(self._call_api_for_negative_response(problem, positive_response, i, semaphore))
                 for i, (problem, positive_response) in enumerate(inputs)]
        total = len(tasks)
        completed = 0
        # Initialize results list with None
        results = [None] * total
        
        pbar = tqdm(desc="Generating negative responses", total=total)

        for future in asyncio.as_completed(tasks):
            request_id, result = await future
            results[request_id] = result 
            completed += 1
            pbar.update(1)
        pbar.close()
        return results

    def compute_position_id_from_first_valid_1d(self, mask):
        """
        Compute position ids based on the first valid token in a 1D mask.
        """
        # Ensure mask is a 1D tensor
        if mask.dim() != 1:
            mask = mask.squeeze()
            if mask.dim() != 1:
                raise ValueError(f"Input mask must be a 1D tensor, but got shape {mask.shape}")

        if mask.sum() == 0:
            return torch.zeros_like(mask, dtype=torch.long)

        seq_len = mask.shape[0]
        
        first_valid_pos = torch.argmax(mask.float())
        
        position_ids = torch.arange(seq_len, device=mask.device, dtype=torch.long) - first_valid_pos
        # Ensure position ids are non-negative
        position_ids = torch.clamp(position_ids, min=0)
        
        return position_ids
