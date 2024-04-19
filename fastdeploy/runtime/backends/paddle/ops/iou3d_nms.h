// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if defined(PADDLEINFERENCE_API_COMPAT_2_4_x)
#include "paddle/include/experimental/ext_all.h"
#elif defined(PADDLEINFERENCE_API_COMPAT_2_5_x)
#include "paddle/include/paddle/extension.h"
#else
#include "paddle/extension.h"
#endif

#include "fastdeploy/utils/utils.h"
#include "fastdeploy/utils/gpu_macro.h"

#if defined(WITH_GPU) || defined(WITH_DCU)
namespace fastdeploy {
namespace paddle_custom_ops {

FASTDEPLOY_DECL int boxes_overlap_bev_gpu(
  paddle::Tensor boxes_a, paddle::Tensor boxes_b,
  paddle::Tensor ans_overlap);
FASTDEPLOY_DECL int boxes_iou_bev_gpu(paddle::Tensor boxes_a, 
                                      paddle::Tensor boxes_b,
                                      paddle::Tensor ans_iou);
FASTDEPLOY_DECL std::vector<paddle::Tensor> nms_gpu(
  const paddle::Tensor& boxes, float nms_overlap_thresh);
FASTDEPLOY_DECL int nms_normal_gpu(
  paddle::Tensor boxes, paddle::Tensor keep, float nms_overlap_thresh);

}  // namespace fastdeploy
}  // namespace paddle_custom_ops

#endif

