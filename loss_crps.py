import torch
import torch.nn.functional as F
from torch import nn
from torch_harmonics import RealSHT
from grids import GridQuadrature
import math


def compute_channel_weighting_helper(channel_names) -> torch.Tensor:
    """
    auxiliary routine for predetermining channel weighting
    """

    # initialize empty tensor
    channel_weights = torch.ones(len(channel_names), dtype=torch.float32)

    for c, chn in enumerate(channel_names):
        if chn in ["u10m", "v10m", "tp"]:
            channel_weights[c] = 0.1
        elif chn in ["t2m", "d2m"]:
            channel_weights[c] = 1.0
        elif chn[0] in ["z", "u", "v", "t", "q"]:
            pressure_level = float(chn[1:])
            channel_weights[c] = 0.001 * pressure_level
        else:
            channel_weights[c] = 0.01

    # normalize
    channel_weights = channel_weights / torch.sum(channel_weights)

    return channel_weights


class FourCastNet3Loss(nn.Module):
    def __init__(self, channel_names, img_shape=(240,240), grid_type="equiangular", lambda_spectral=0.1, use_fair_crps=False):
        """
        FourCastNet 3 损失函数实现
        
        Args:
            channel_weights (torch.Tensor): 通道权重，形状为 [C]
            temporal_weights (torch.Tensor): 时间步权重，形状为 [T]
            lambda_spectral (float): 谱损失权重系数
            use_fair_crps (bool): 是否使用公平CRPS计算
        """
        super().__init__()
        channel_names = ["t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000", "t2m", "d2m", "u10m", "v10m", "tp"]
        self.channel_weights = compute_channel_weighting_helper(channel_names).cuda()
        self.lambda_spectral = lambda_spectral
        self.use_fair_crps = use_fair_crps
        
        self.sht = RealSHT(*img_shape, grid=grid_type).cuda().float()
        lmax = self.sht.lmax
        l_weights = torch.ones(lmax).reshape(-1, 1)
        self.register_buffer("l_weights", l_weights, persistent=False)
        mmax = self.sht.mmax
        m_weights = 2 * torch.ones(mmax).reshape(1, -1)
        m_weights[0, 0] = 1.0
        self.register_buffer("m_weights", m_weights, persistent=False)
        
        self.quadrature = GridQuadrature(
            "naive",
            img_shape=img_shape,
            crop_shape=None,
            crop_offset=(0,0),
            normalize=True,
            pole_mask=None,
            distributed=False,
        )
        quad_weight_split = self.quadrature.quad_weight.reshape(1, 1, -1)
        quad_weight_split = quad_weight_split.contiguous()
        self.register_buffer("quad_weight_split", quad_weight_split, persistent=False)
        
    def spatial_crps(self, ensemble, target):
        """
        计算空间CRPS损失
        
        Args:
            ensemble (torch.Tensor): 预测集合，形状为 [B, E, C, H, W]
            target (torch.Tensor): 目标值，形状为 [B, C, H, W]
            spatial_weights (torch.Tensor): 空间积分权重，形状为 [H, W]
            
        Returns:
            torch.Tensor: 空间CRPS损失值
        """
        B, E, C, H, W = ensemble.shape
        
        # 重塑张量以便计算
        ensemble_flat = ensemble.reshape(B, E, C, H * W)
        target_flat = target.reshape(B, C, H * W).unsqueeze(1)  # [B, 1, C, H*W]
        
        # 计算点-wise CRPS
        if self.use_fair_crps:
            crps_values = self.fair_crps(ensemble_flat, target_flat)
        else:
            crps_values = self.biased_crps(ensemble_flat, target_flat) # [B, C, H*W]
        
        # 应用空间权重并积分
        weighted_crps = crps_values * self.quad_weight_split.cuda()
        spatial_integral = weighted_crps.sum(dim=-1)  # 积分 over spatial domain

        return spatial_integral
    
    def spectral_crps(self, ensemble, target):
        """
        计算谱CRPS损失
        
        Args:
            ensemble (torch.Tensor): 预测集合，形状为 [B, E, C, H, W]
            target (torch.Tensor): 目标值，形状为 [B, C, H, W]
            
        Returns:
            torch.Tensor: 谱CRPS损失值
        """
        
        # 应用球谐变换到谱域
        ensemble_spec = self.sht(ensemble.float()) / 4.0 / math.pi  # [B, E, C, L, M]
        target_spec = self.sht(target.unsqueeze(1).float()) / 4.0 / math.pi  # [B, 1, C, L, M]
        
        ensemble_spec = torch.abs(ensemble_spec)
        target_spec = torch.abs(target_spec)
        
        # 计算每个谱系数的CRPS
        if self.use_fair_crps:
            crps_values = self.fair_crps(ensemble_spec, target_spec)
        else:
            crps_values = self.biased_crps(ensemble_spec, target_spec) # [B, C, H*W]
        
        B, C, H, W = crps_values.shape
        crps_values = crps_values.reshape(B, C, H*W)
        spectral_weights = self.m_weights * self.l_weights
        spectral_weights_split = spectral_weights.cuda().reshape(1, 1, H * W)
        # 对所有谱系数求和
        spectral_sum = torch.sum(crps_values * spectral_weights_split, dim=-1)
        
        return spectral_sum
    
    def biased_crps(self, ensemble, target):
        """
        有偏CRPS计算（公式46）
        
        Args:
            ensemble (torch.Tensor): 预测集合，形状为 [..., E]
            target (torch.Tensor): 目标值，形状为 [..., 1]
            
        Returns:
            torch.Tensor: CRPS值，形状与target相同但去掉E维度
        """
        # 实现有偏CRPS计算公式
        term1 = torch.abs(ensemble - target).mean(dim=1)  # E[|U - u*|]
        term2 = 0.5 * torch.abs(ensemble.unsqueeze(2) - ensemble.unsqueeze(1)).mean(dim=(1, 2))  # 0.5E[|U - U'|]
        
        return term1 - term2
    
    def fair_crps(self, ensemble, target):
        """
        公平CRPS计算（公式47）
        
        Args:
            ensemble (torch.Tensor): 预测集合，形状为 [..., E]
            target (torch.Tensor): 目标值，形状为 [..., 1]
            
        Returns:
            torch.Tensor: CRPS值，形状与target相同但去掉E维度
        """
        # 实现公平CRPS计算公式
        term1 = torch.abs(ensemble - target).mean(dim=1)  # E[|U - u*|]
        
        # 计算所有配对差异的绝对值
        diff_matrix = torch.abs(ensemble.unsqueeze(2) - ensemble.unsqueeze(1))
        mask = ~torch.eye(ensemble.size(1), dtype=torch.bool, device=ensemble.device)
        term2 = 0.5 * diff_matrix[:, mask].mean(dim=1)  # 0.5E[|U - U'|] 排除自配对
        
        return term1 - term2
    
    def forward(self, ensemble, target):
        """
        计算完整损失函数
        
        Args:
            ensemble (torch.Tensor): 预测集合，形状为 [B, T, E, C, H, W]
            target (torch.Tensor): 目标值，形状为 [B, T, C, H, W]
            spatial_weights (torch.Tensor): 空间积分权重，形状为 [H, W]
            
        Returns:
            torch.Tensor: 总损失值
        """
        B, T, E, C, H, W = ensemble.shape
        
        # 初始化损失
        total_loss = 0.0
        
        # 遍历时间步
        for t in range(T):
            # 获取当前时间步的预测和目标
            ensemble_t = ensemble[:, t]  # [B, E, C, H, W]
            target_t = target[:, t]  # [B, C, H, W]
            
            # 计算空间CRPS损失
            spatial_loss = self.spatial_crps(ensemble_t, target_t)
            
            # 计算谱CRPS损失
            spectral_loss = self.spectral_crps(ensemble_t, target_t)
            
            # 结合空间和谱损失
            combined_loss = spatial_loss + self.lambda_spectral * spectral_loss #b,c
            
            # 应用通道权重
            channel_weighted = combined_loss * self.channel_weights.unsqueeze(0)
            
            # 求和 over channels
            channel_sum = channel_weighted.sum(dim=1)
            
            # 应用时间步权重并累加
            total_loss += channel_sum.mean()  # 平均 over batch
        
        return total_loss / T