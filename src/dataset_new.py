# 创建完整的dataset.py文件
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class IceThicknessDataset(Dataset):
    """
    覆冰厚度预测数据集类
    
    数据结构：
    - 每个CSV文件包含79条记录
    - 索引0-5：6条历史状态数据（已知）
    - 索引6：1条当前观测数据（已知）
    - 索引7-78：72条未来预测数据（用于生成标签）
    
    输入特征：历史数据 + 当前数据的数值特征
    输出标签：未来72小时的覆冰厚度_期望（可预测任一小时）
    """
    
    def __init__(self, 
                 data_dir: str, 
                 feature_cols: Optional[List[str]] = None,
                 target_col: str = "覆冰厚度_期望",
                 normalize: bool = True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据文件所在目录路径
            feature_cols: 用于训练的特征列名列表，默认使用所有数值特征（除终端编号和时间）
            target_col: 目标列名，默认"覆冰厚度_期望"
            normalize: 是否对特征和标签进行标准化
            mean_std_path: 均值标准差文件路径（用于测试集标准化）
        """
        # 初始化参数
        self.data_dir = data_dir
        self.target_col = target_col
        self.normalize = normalize

        last_dir_name = os.path.basename(os.path.normpath(self.data_dir))
        mean_std_path = last_dir_name+"_"+"data_mean_std.npz"
        
        # 获取所有CSV数据文件
        self.file_paths = self._get_data_files()
        if not self.file_paths:
            raise ValueError(f"在目录 {data_dir} 中未找到任何CSV文件")
        
        # 定义特征列（默认使用所有数值特征）
        if feature_cols is None:
            self.feature_cols = [
                "覆冰厚度_期望", "覆冰厚度_标准差", "覆冰厚度_数据条数",
                "覆冰比值_期望", "覆冰比值_标准差", "覆冰比值_数据条数",
                "温度_期望", "温度_标准差", "温度_数据条数",
                "湿度_期望", "湿度_标准差", "湿度_数据条数"
            ]
        else:
            self.feature_cols = feature_cols
        
        # 计算或加载数据标准化参数
        if self.normalize:
            if mean_std_path and os.path.exists(mean_std_path):
                self._load_mean_std(mean_std_path)
            else:
                self._compute_mean_std()
                # 保存均值标准差到文件（如果提供了路径）
                if mean_std_path:
                    self._save_mean_std(mean_std_path)
    
    def _get_data_files(self) -> List[str]:
        """递归获取父目录下所有子文件夹中的CSV数据文件"""
        file_paths = []
        # 递归遍历目录树（父目录 + 所有子目录）
        for root, dirs, files in os.walk(self.data_dir):
            for filename in files:
                # 筛选条件：CSV文件 + 文件名包含"覆冰厚度"
                if filename.endswith(".csv") and "覆冰厚度" in filename:
                    # 拼接完整文件路径
                    full_path = os.path.join(root, filename)
                    file_paths.append(full_path)
        # 按文件路径排序（保证顺序一致性）
        return sorted(file_paths)
    
    def _compute_mean_std(self) -> None:
        """计算所有数据的均值和标准差用于标准化"""
        all_features = []
        
        # 遍历所有文件收集特征数据
        for file_path in self.file_paths:
            df = self._load_single_file(file_path)
            
            # 收集历史数据（0-5）和当前数据（6）的特征
            historical_features = df.iloc[0:6][self.feature_cols].values
            current_features = df.iloc[6][self.feature_cols].values.reshape(1, -1)
            
            # 合并历史和当前特征
            sample_features = np.vstack([historical_features, current_features])
            all_features.append(sample_features)
        
        # 计算整体均值和标准差
        all_features_np = np.concatenate(all_features, axis=0)

        # ========== 关键修复：彻底清理数据 ==========
        # 1. 强制转为float32（核心！解决object类型问题）
        all_features_np = all_features_np.astype(np.float32)
        # 2. 替换NaN/inf为有效值（避免计算报错）
        all_features_np = np.nan_to_num(
            all_features_np,
            nan=0.0,       # NaN替换为0
            posinf=1e6,    # 正无穷替换为1e6
            neginf=-1e6    # 负无穷替换为-1e6
        )
        
        # ========== 调试代码（可选，验证修复） ==========
        print("修复后数据检查：")
        print(f"1. 数组形状：{all_features_np.shape}")
        print(f"2. 数组类型：{all_features_np.dtype}")  # 必须是float32/float64
        print(f"3. 是否包含NaN：{np.isnan(all_features_np).any()}")  # 现在能正常运行
        print(f"4. 是否包含inf：{np.isinf(all_features_np).any()}")

        self.feature_mean = np.mean(all_features_np, axis=0)
        self.feature_std = np.std(all_features_np, axis=0)
        
        # 避免标准差为0的情况
        self.feature_std[self.feature_std < 1e-6] = 1e-6
        
        print(f"数据标准化参数计算完成：")
        print(f"特征均值形状: {self.feature_mean.shape}")
        print(f"特征标准差形状: {self.feature_std.shape}")
    
    def _load_mean_std(self, path: str) -> None:
        """从文件加载均值和标准差"""
        data = np.load(path)
        self.feature_mean = data["mean"]
        self.feature_std = data["std"]
        
        # 验证维度是否匹配
        if len(self.feature_mean) != len(self.feature_cols):
            raise ValueError(f"加载的均值维度 {len(self.feature_mean)} 与特征数量 {len(self.feature_cols)} 不匹配")
        
        print(f"从 {path} 加载标准化参数完成")
    
    def _save_mean_std(self, path: str) -> None:
        """保存均值和标准差到文件"""
        np.savez(path, mean=self.feature_mean, std=self.feature_std)
        print(f"标准化参数已保存到 {path}")
    
    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """加载单个CSV文件并强制特征列为数值型"""
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 验证数据条数
        if len(df) != 79:
            raise ValueError(f"文件 {os.path.basename(file_path)} 数据条数异常：{len(df)}（需79条）")
        
        # 核心修复：强制所有特征列转为数值型，无法转换的设为NaN
        for col in self.feature_cols:
            # 1. 先替换常见的非数值字符串（如"无数据"、"NA"、空字符串）
            df[col] = df[col].replace(['', 'NA', '无数据', 'nan', 'NaN'], np.nan)
            # 2. 强制转换为float，失败则设为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 3. 填充NaN（用列均值，也可根据业务用0）
            if df[col].isnull().sum() > 0:
                col_mean = df[col].mean()
                df[col] = df[col].fillna(col_mean)
                print(f"警告：{os.path.basename(file_path)} 的 {col} 列填充了 {df[col].isnull().sum()} 个NaN")
        
        return df
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """对特征进行标准化处理"""
        if not self.normalize:
            return features
        
        # 1. 强制转为float32（核心！解决object类型问题）
        features = features.astype(np.float32)
        # 2. 替换NaN/inf为有效值（避免计算报错）
        features = np.nan_to_num(
            features,
            nan=0.0,       # NaN替换为0
            posinf=1e6,    # 正无穷替换为1e6
            neginf=-1e6    # 负无穷替换为-1e6
        )
        #return (features - self.feature_mean) / self.feature_std
        # 只对温度及湿度进行标准化
        features[:,6] = (features[:,6]-self.feature_mean[6]) / self.feature_std[6]
        features[:,9] = (features[:,9]-self.feature_mean[9]) / self.feature_std[9]
        return features

    
    def __len__(self) -> int:
        """返回数据集样本数量（即CSV文件数量）"""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            features: 输入特征 (7, num_features) - 6条历史 + 1条当前
            targets: 目标标签 (72,) - 未来72小时的覆冰厚度_期望
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"索引 {idx} 超出数据集范围 [0, {len(self)-1}]")
        
        # 加载对应的CSV文件
        file_path = self.file_paths[idx]
        df = self._load_single_file(file_path)
        
        # 提取输入特征：6条历史数据（0-5） + 1条当前数据（6）
        historical_data = df.iloc[0:6][self.feature_cols].values  # (6, num_features)
        current_data = df.iloc[6][self.feature_cols].values.reshape(1, -1)  # (1, num_features)
        
        # 合并特征并标准化
        raw_features = np.vstack([historical_data, current_data])  # (7, num_features)
        normalized_features = self._normalize_features(raw_features)

        # 未来72小时温度数据
        temp_features_72 = df.iloc[7:79]['温度_期望'].values
        temp_features_72 = (temp_features_72-self.feature_mean[6]) / self.feature_std[6]

        # 未来72小时湿度数据
        humid_features_72 = df.iloc[7:79]['湿度_期望'].values 
        humid_features_72 = (humid_features_72-self.feature_mean[9]) / self.feature_mean[9]
        
        # 提取目标标签：未来72小时的覆冰厚度_期望（7-78）
        targets = df.iloc[7:79][self.target_col].values  # (72,)
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(normalized_features)
        targets_tensor = torch.FloatTensor(targets)

        temp_features_72_tensor = torch.FloatTensor(temp_features_72)
        humid_features_72_tensor = torch.FloatTensor(humid_features_72)
        
        return features_tensor, targets_tensor, temp_features_72_tensor, humid_features_72_tensor


def create_data_loaders(train_dir: str, 
                        val_dir: Optional[str] = None,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        pin_memory: bool = True,
                        normalize: bool = True,
                        mean_std_path: str = "data_mean_std.npz") -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建训练和验证数据加载器
    
    Args:
        train_dir: 训练数据目录
        val_dir: 验证数据目录（可选）
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用固定内存（加速GPU训练）
        normalize: 是否进行数据标准化
        mean_std_path: 均值标准差文件路径
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（如果提供了val_dir）
    """
    # 验证训练目录是否存在
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练数据目录 {train_dir} 不存在")
    
    last_dir_name = os.path.basename(os.path.normpath(train_dir))
    mean_std_path = last_dir_name+"_"+"data_mean_std.npz"


    # 创建训练数据集（计算标准化参数）
    train_dataset = IceThicknessDataset(
        data_dir=train_dir,
        normalize=normalize
    )
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # 不丢弃最后一个不完整批次
    )
    
    print(f"训练数据加载器创建完成：")
    print(f"训练样本数量: {len(train_dataset)}")
    print(f"训练批次数量: {len(train_loader)}")
    print(f"批次大小: {batch_size}")
    
    # 创建验证数据加载器（如果提供了验证目录）
    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = IceThicknessDataset(
            data_dir=val_dir,
            normalize=normalize,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证集不打乱
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        print(f"\\n验证数据加载器创建完成：")
        print(f"验证样本数量: {len(val_dataset)}")
        print(f"验证批次数量: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    """测试数据集和数据加载器功能"""
    '''
    import argparse
    
    parser = argparse.ArgumentParser(description="覆冰厚度数据集测试")
    parser.add_argument("--data_dir", type=str, required=True, help="数据文件目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    
    args = parser.parse_args()
    '''
    
    # 测试数据集
    print("="*50)
    print("测试数据集初始化...")
    data_dir = r'D:\work\DianLi\IcethickRegression\data\data\分片数据_前6_当前_后72\CC3988'
    batch_size = 32
    num_workers = 4

    dataset = IceThicknessDataset(
        data_dir=data_dir,
        normalize=True
    )
    
    print(f"\\n数据集样本数量: {len(dataset)}")
    
    # 测试单个样本加载
    if len(dataset) > 0:
        print("\\n" + "="*50)
        print("测试单个样本加载...")
        features, targets, temps_72, humids_72 = dataset[0]
        
        print(f"输入特征形状: {features.shape} (7条数据 × {features.shape[1]}个特征)")
        print(f"未来72小时温度形状: {temps_72.shape}")
        print(f"未来72小时湿度形状: {humids_72.shape}")
        print(f"目标标签形状: {targets.shape} (未来72小时)")
        print(f"输入特征前2条数据:")
        print(features[:2])
        print(f"目标标签前10个值:")
        print(targets[:10])
    
    # 测试数据加载器
    if len(dataset) > 0:
        print("\\n" + "="*50)
        print("测试数据加载器...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # 迭代一个批次
        for batch_idx, (batch_features, batch_targets, batch_temps_72, batch_humids_72) in enumerate(dataloader):
            print(f"批次 {batch_idx+1}:")
            print(f"  批次特征形状: {batch_features.shape} (batch_size × 7 × num_features)")
            print(f"  批次标签形状: {batch_targets.shape} (batch_size × 72)")
            print(f"  批次未来72小时温度形状: {batch_temps_72.shape} (batch_size × 72)")
            print(f"  批次未来72小时湿度形状: {batch_humids_72.shape} (batch_size × 72)")
            
            # 只测试一个批次
            if batch_idx == 100:
                break
    
    print("\\n" + "="*50)
    print("所有测试完成！")
