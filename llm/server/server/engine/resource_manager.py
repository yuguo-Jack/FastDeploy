# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import os
import random
import threading
import time

import numpy as np
from server.utils import model_server_logger


class ResourceManager(object):
    """
    用于记录和分配引擎的资源
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.stop_flags = [True] * cfg.max_batch_size
        self.free_list = list(range(cfg.max_block_num - 1, -1, -1))
        self.tasks_list = [None] * self.cfg.max_batch_size
        # 引擎当前的batch情况
        self.real_bsz = 0
        model_server_logger.info(f"{self.info()}")

    def get_required_block_number(self, input_token_num):
        """
        计算需要多少Block资源
        """
        block_num = (input_token_num + self.cfg.block_size - 1 + self.cfg.dec_token_num) // self.cfg.block_size
        return block_num

    def get_encoder_block_number(self, input_token_num):
        """
        获取编码器所需的block数目
        """
        enc_block_num = (input_token_num + self.cfg.block_size - 1) // self.cfg.block_size
        return enc_block_num

    def get_decoder_block_number(self):
        """
        获取解码器所需的block数目
        """
        return (self.cfg.dec_token_num + self.cfg.block_size - 1) // self.cfg.block_size

    def total_block_number(self):
        """
        返回服务启动时预分配的block数量
        """
        return self.cfg.max_block_num

    def _get_block_tables(self, input_token_num, required_type="all"):
        """
        分配显存资源
        """
        if required_type == "all":
            block_num = self.get_required_block_number(input_token_num)
        elif required_type == "encoder":
            block_num = self.get_encoder_block_number(input_token_num)
        elif required_type == "decoder":
            block_num = self.get_decoder_block_number()
        else:
            raise ValueError('unknown required type')
        block_num = min(block_num, self.cfg.max_query_block_num)
        block_list = list()
        if block_num > len(self.free_list):
            model_server_logger.error("block_num:{0} > free_list len:{1}".format(block_num, len(self.free_list)))
            return block_list
        for _ in range(block_num):
            used_block_id = self.free_list.pop()
            block_list.append(used_block_id)
        model_server_logger.info(f"dispatch {len(block_list)} blocks.")
        return block_list

    def _recycle_block_tables(self, block_tables):
        """
        回收显存资源blocks
        """
        ori_number = len(self.free_list)
        self.free_list.extend(block_tables)
        # self.free_list = list(set(self.free_list + block_tables))
        cur_number = len(self.free_list)
        model_server_logger.info(f"recycle {cur_number - ori_number} blocks.")

    def available_batch(self):
        """
        引擎当前可用最大Batch
        """
        return np.sum(self.stop_flags)

    def availabel_block_num(self):
        """
        引擎当前可用的block数量
        """
        return len(self.free_list)

    def is_resource_sufficient(self, input_token_num):
        """
        判断当前可用资源是否满足新的需求
        """
        if self.available_batch() < 1:
            return False
        block_num = self.get_required_block_number(input_token_num)
        if block_num > self.availabel_block_num():
            return False
        return True

    def allocate_resources_for_new_tasks(self, tasks):
        """
        为新任务分配资源
        """

        allocated_position = 0 # 新任务插入的位置
        processing_task_index = 0 # 当前正在处理的任务index
        processed_tasks = list()
        while allocated_position < self.cfg.max_batch_size:
            if processing_task_index >= len(tasks):
                break

            if len(tasks[processing_task_index]["input_ids"]) > self.cfg.max_seq_len:
                model_server_logger.error("req_id: {0} input_ids len:{1} > {2}".format(
                    tasks[
                        processing_task_index]["req_id"], len(tasks[
                        processing_task_index]["input_ids"]), self.cfg.max_seq_len
                ))
                processing_task_index += 1
                continue

            can_insert = False
            while allocated_position + 1 <= self.cfg.max_batch_size:
                if sum(self.stop_flags[allocated_position : allocated_position + 1]) == 1:
                    can_insert = True
                    break
                allocated_position += 1
            if can_insert:
                if self.stop_flags[allocated_position]:
                    task = copy.deepcopy(tasks[processing_task_index])

                    if not isinstance(task["eos_token_ids"], list):
                        task["eos_token_ids"] = [task["eos_token_ids"]]

                    if "infer_seed" in task and task["infer_seed"]:
                        task["infer_seed"] = int(task["infer_seed"])
                    else:
                        task["infer_seed"] = random.randint(0, 9223372036854775807)
                    task["idx"] = allocated_position
                    task["block_tables"] = self._get_block_tables(len(task["input_ids"]))
                    if not task["block_tables"]:
                        model_server_logger.error("req_id: {0} block_tables is empty".format(task["req_id"]))
                        continue

                    processed_tasks.append(task)
                    self.stop_flags[allocated_position] = False
                    task["inference_start_time"] = time.time()
                    task["inference_time_cost"] = -1.0
                    task["tokens_all_num"] = int(0)
                    self.tasks_list[allocated_position] = task
                    model_server_logger.info(f"allocate req_id: {task['req_id']}, "
                                            f"allocated_position:{allocated_position}, input_ids_length: {len(task['input_ids'])}")
                allocated_position += 1
            processing_task_index += 1

        # 统计引擎正在推理时的batch size
        for i in range(self.cfg.max_batch_size - 1, -1, -1):
            if not self.stop_flags[i]:
                self.real_bsz = i + 1
                break

        model_server_logger.info("in num:{0} new task num:{1} real_bsz is:{2}".format(
            len(tasks), len(processed_tasks), self.real_bsz))
        model_server_logger.info(f"{self.info()}")
        return processed_tasks

    def info(self):
        info = f"ResourceManager info, " \
               f"total_block_number: {self.total_block_number()}, total_batch_number: {len(self.stop_flags)}, " \
               f"availabel_block_num: {self.availabel_block_num()}, available_batch: {self.available_batch()}"
        return info
