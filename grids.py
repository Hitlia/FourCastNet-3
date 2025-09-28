# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from torch_harmonics.quadrature import legendre_gauss_weights


def grid_to_quadrature_rule(grid_type):

    grid_to_quad_dict = {"euclidean" : "uniform", "equiangular" : "naive", "legendre-gauss" : "legendre-gauss", "clenshaw-curtiss" : "clenshaw-curtiss", "weatherbench2" : "weatherbench2"}

    if grid_type not in grid_to_quad_dict.keys():
        raise NotImplementedError(f"Grid type {grid_type} does not have a quadrature rule")
    else:
        return grid_to_quad_dict[grid_type]


class GridConverter(torch.nn.Module):
    def __init__(self, src_grid, dst_grid, lat_rad, lon_rad):
        super(GridConverter, self).__init__()
        self.src = src_grid
        self.dst = dst_grid
        self.src_lat = lat_rad
        self.src_lon = lon_rad
        self.dst_lat = self.src_lat
        self.dst_lon = self.src_lon

    def get_src_coords(self):
        return self.src_lat, self.src_lon

    def get_dst_coords(self):
        return self.dst_lat, self.dst_lon

    def forward(self, data):
        if self.src == self.dst:
            return data
        else:
            return torch.lerp(data[..., self.indices, :], data[..., self.indices + 1, :], self.interp_weights.to(dtype=data.dtype))


class GridQuadrature(torch.nn.Module):
    def __init__(self, quadrature_rule, img_shape, crop_shape=None, crop_offset=(0, 0), normalize=False, pole_mask=None, distributed=False):
        super().__init__()

        crop_shape = img_shape if crop_shape is None else crop_shape

        if quadrature_rule == "naive":
            jacobian = torch.clamp(torch.sin(torch.linspace(0, torch.pi, 721)), min=0.0)
            dtheta = torch.pi / 721#img_shape[0]
            dlambda = 2 * torch.pi / 1440# img_shape[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian.unsqueeze(1)
            quad_weight = quad_weight.tile(1, 1440)
            # numerical precision can be an issue here, make sure it sums to 4pi:
            quad_weight = quad_weight * (4.0 * torch.pi) / torch.sum(quad_weight)
            quad_weight = torch.flip(quad_weight[361:361+img_shape[0],:img_shape[1]],dims=[0])
        elif quadrature_rule == "legendre-gauss":
            cost, weights = legendre_gauss_weights(img_shape[0], -1, 1)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * weights.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # apply pole mask
        # if (pole_mask is not None) and (pole_mask > 0):
        #     quad_weight[:pole_mask, :] = 0.0
        #     quad_weight[sizes[0] - pole_mask :, :] = 0.0

        # if distributed, make sure to split correctly across ranks:
        # in case of model parallelism, we need to make sure that we use the correct shapes per rank
        # for h
        local_shape_h = crop_shape[0]
        local_offset_h = crop_offset[0]

        # for w
        local_shape_w = crop_shape[1]
        local_offset_w = crop_offset[1]

        # crop globally if requested
        if crop_shape is not None:
            quad_weight = quad_weight[local_offset_h : local_offset_h + local_shape_h, local_offset_w : local_offset_w + local_shape_w]

        # make it contiguous
        quad_weight = quad_weight.contiguous()

        # reshape
        H, W = quad_weight.shape
        quad_weight = quad_weight.reshape(1, 1, H, W)

        self.register_buffer("quad_weight", quad_weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # integrate over last two axes only:
        quad = torch.sum(x * self.quad_weight, dim=(-2, -1))
        return quad
