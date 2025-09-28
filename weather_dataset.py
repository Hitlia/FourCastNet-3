import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime

# 导入天顶角计算函数
from zenith_angle import cos_zenith_angle

class WeatherPanguData(Dataset):
    def __init__(self, timestamps, npy_path, tp6hr_path=None, input_window_size=1, output_window_size=1):
        """
        Args:
            timestamps: 按时间顺序排列的 datetime 对象列表，间隔为6小时
            npy_path: 存放 npy 文件的根目录，比如 "npy_path"
            lat_grid: 纬度网格 (H, W)
            lon_grid: 经度网格 (H, W)
            input_window_size: 输入窗口大小（时间步数）
            output_window_size: 目标窗口大小（时间步数）
        """
        self.timestamps = timestamps
        self.npy_path = npy_path
        self.tp6hr_path = tp6hr_path
        
        longitude = np.arange(80, 140, 0.25)
        latitude = np.arange(60, 0, -0.25)
        self.lon_grid, self.lat_grid = np.meshgrid(longitude, latitude)
        
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.total_window = input_window_size + output_window_size
        # 每个样本需要 total_window 个时间点，因此总样本数如下：
        self.num_samples = len(self.timestamps) - self.total_window + 1 

    def __len__(self):
        return self.num_samples

    def _get_filepath(self, timestamp):
        """
        根据时间戳构造 npy 文件路径，文件名格式：
        npy_path/YYYY/YYYY_MM_DD_HH.npy
        例如：2016年1月1日1点 --> npy_path/2016/2016_01_01_01.npy
        """ 
        year = timestamp.year
        file_name = f"{timestamp.year}_{timestamp.month:02d}_{timestamp.day:02d}_{timestamp.hour:02d}.npy"
        file_path = os.path.join(self.npy_path, str(year), file_name)
        return file_path
    
    def _get_tp6hr_filepath(self, timestamp):
        year = timestamp.year
        file_name = f"{timestamp.year}_{timestamp.month:02d}_{timestamp.day:02d}_{timestamp.hour:02d}.npy"
        file_path = os.path.join(self.tp6hr_path, str(year), file_name)
        return file_path

    def __getitem__(self, idx):
        # 选取输入窗口和目标窗口对应的时间戳序列
        input_timestamps = self.timestamps[idx: idx + self.input_window_size]
        target_timestamps = self.timestamps[idx + self.input_window_size: idx + self.total_window]

        input_list = []
        target_list = []
        input_timestamps_np = np.array(input_timestamps)
        target_timestamps_np = np.array(target_timestamps)
        
        if self.tp6hr_path is not None:
            for ts in input_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :240, :240]                # [70, 241, 241]
                tp6hr_file = self._get_tp6hr_filepath(ts)
                tp6hr_data = np.load(tp6hr_file)[:, :240, :240]
                data[69, :, :] = tp6hr_data[0, :, :]
                input_list.append(data)
            for ts in target_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :240, :240]
                tp6hr_file = self._get_tp6hr_filepath(ts)
                tp6hr_data = np.load(tp6hr_file)[:, :240, :240]
                data[69, :, :] = tp6hr_data[0, :, :]
                target_list.append(data)
        else:
            # 读取气象数据
            for ts in input_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :240, :240]  # [70, 240, 240]
                input_list.append(data)
                
            for ts in target_timestamps:
                file = self._get_filepath(ts)
                data = np.load(file)[:70, :240, :240]
                target_list.append(data)

        # 转换为张量
        input_array = np.stack(input_list, axis=0)  # [input_window_size, 70, 240, 240]
        input_tensor = torch.from_numpy(input_array).float()
        input_tensor = torch.nan_to_num(input_tensor)

        target_array = np.stack(target_list, axis=0)
        target_tensor = torch.from_numpy(target_array).float()
        target_tensor = torch.nan_to_num(target_tensor)  # [output_window_size, 70, 240, 240]

        # 计算天顶角
        input_zenith = cos_zenith_angle(input_timestamps_np, self.lon_grid, self.lat_grid)
        target_zenith = cos_zenith_angle(target_timestamps_np, self.lon_grid, self.lat_grid)
        
        # 转换为张量并调整形状
        input_zenith_tensor = torch.from_numpy(input_zenith).float().unsqueeze(1)  # [input_window_size, 1, 240, 240]
        target_zenith_tensor = torch.from_numpy(target_zenith).float().unsqueeze(1)  # [output_window_size, 1, 240, 240]

        # 转换为时间戳（Unix时间戳）
        input_timestamps_unix = np.array([ts.timestamp() for ts in input_timestamps])
        target_timestamps_unix = np.array([ts.timestamp() for ts in target_timestamps])
        
        input_timestamps_tensor = torch.from_numpy(input_timestamps_unix).float()
        target_timestamps_tensor = torch.from_numpy(target_timestamps_unix).float()

        batch_data = {
            'input': input_tensor,  # 形状: [input_window_size, 70, H, W]
            'target': target_tensor,  # 形状: [output_window_size, 70, H, W]
            'input_zenith': input_zenith_tensor,  # 形状: [input_window_size, 1, H, W]
            'target_zenith': target_zenith_tensor,  # 形状: [output_window_size, 1, H, W]
            'input_timestamps': input_timestamps_tensor,  # 形状: [input_window_size]
            'target_timestamps': target_timestamps_tensor,  # 形状: [output_window_size]
            'input_datetime': [ts.strftime('%Y-%m-%dT%H:%M:%S') for ts in input_timestamps],
            'target_datetime': [ts.strftime('%Y-%m-%dT%H:%M:%S') for ts in target_timestamps],
        }
        return batch_data