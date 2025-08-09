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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import ray
import time


def main() -> None:
    # if not ray.is_initialized():
    #     # this is for local ray cluster
    #     ray.init(
    #         runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}, namespace='verl')
    
    sync = ray.get_actor("SyncServer", namespace="verl")

    while True:
        should_stop = ray.get(sync.should_stop.remote())
        print(f"Should Stop: {should_stop}")
        if should_stop:
            break
        else:
            print("Main process does not exist. Keep sleeping...", flush=True)
            time.sleep(60)

    print("Null node exiting gracefully...")


if __name__ == '__main__':
    main()
