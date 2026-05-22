"""
模型评估脚本 — 加载best.ckpt评估Seq2ABTransformer性能
需在有GPU和xFormers的环境上运行

输出：
  1. 预测结果文件（predictions.npz）
  2. 评估指标（MAE, RMSE, Accuracy@多阈值）
  3. 评估报告（eval_report.txt）

使用方法：
  python experiments/evaluate.py
"""

import torch
import numpy as np
import os
import sys
import json
from copy import deepcopy

# 添加项目根目录到path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import FubingDataset
from src.model import Seq2ABTransformer

# ========== 配置 ==========
cfg = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'batch_size': 64,
    'cdt_length': 2 * 24 * 10 * 3,  # 1440
    'weights_path': os.path.join(PROJECT_ROOT, 'weights', '20251205', 'best.ckpt'),
    'output_dir': os.path.join(PROJECT_ROOT, 'experiments', 'results'),
    'test_size': 2000,  # 测试样本数
    'max_pred_window': 2 * 24 * 10,  # 最多预测48小时内的点
}


@torch.no_grad()
def evaluate_model(model, dataset, cfg):
    """全面评估模型性能"""
    model.eval()
    dataset.indices = list(range(dataset.total_length))  # 不打乱

    all_preds = []      # 预测值
    all_labels = []     # 真实值
    all_errors = []     # 绝对误差
    all_time_steps = [] # 预测的时间步（用于分析不同预测窗口的精度）
    
    count = 0
    skip_count = 0
    
    print(f"开始评估... 测试样本数: {cfg['test_size']}")
    
    for i, (batch_pre, batch_post) in enumerate(dataset):
        if count >= cfg['test_size']:
            break
        
        for j in range(len(batch_pre)):
            if count >= cfg['test_size']:
                break
                
            pre_data = torch.from_numpy(batch_pre[j]).to(cfg['device'])[None]
            post_data = batch_post[j]
            
            if len(post_data) == 0:
                skip_count += 1
                continue
            
            max_idx = min(len(post_data), cfg['max_pred_window'])
            
            # 对每个样本测试多个预测时间点
            test_indices = np.linspace(0, max_idx - 1, min(10, max_idx)).astype(int)
            
            for select_idx in test_indices:
                label = torch.from_numpy(post_data[select_idx, :2]).to(cfg['device'])[None]
                _cdt = torch.from_numpy(post_data[:select_idx + 1, -3:].reshape(1, -1)).to(cfg['device'])
                
                if _cdt.shape[1] > cfg['cdt_length']:
                    continue
                
                cdt = torch.cat([_cdt, torch.zeros(1, cfg['cdt_length'] - _cdt.shape[1]).to(cfg['device'])], dim=1)
                
                logit = model(pre_data, cdt)
                
                pred_val = logit[0, 0].item()
                true_val = label[0, 0].item()
                error = abs(pred_val - true_val)
                
                all_preds.append(logit[0].cpu().numpy())
                all_labels.append(label[0].cpu().numpy())
                all_errors.append(error)
                all_time_steps.append(select_idx)
            
            count += 1
            
            if count % 200 == 0:
                print(f"  已评估: {count}/{cfg['test_size']} 样本")
    
    print(f"评估完成: 有效样本 {count}, 跳过 {skip_count}")
    
    all_preds = np.array(all_preds)     # (N, 2)
    all_labels = np.array(all_labels)   # (N, 2)
    all_errors = np.array(all_errors)   # (N,)
    all_time_steps = np.array(all_time_steps)
    
    return all_preds, all_labels, all_errors, all_time_steps


def compute_metrics(preds, labels, errors):
    """计算全面的评估指标"""
    # 覆冰厚度（第0列）
    pred_thickness = preds[:, 0]
    true_thickness = labels[:, 0]
    
    mae = np.mean(np.abs(pred_thickness - true_thickness))
    rmse = np.sqrt(np.mean((pred_thickness - true_thickness) ** 2))
    
    # 多阈值准确率
    thresholds = [0.1, 0.2, 0.3, 0.5, 1.0]
    accuracies = {}
    for thresh in thresholds:
        acc = np.mean(np.abs(pred_thickness - true_thickness) < thresh)
        accuracies[f'Acc@{thresh}'] = acc
    
    # 覆冰比值（第1列）
    pred_ratio = preds[:, 1]
    true_ratio = labels[:, 1]
    mae_ratio = np.mean(np.abs(pred_ratio - true_ratio))
    rmse_ratio = np.sqrt(np.mean((pred_ratio - true_ratio) ** 2))
    
    # R² score
    ss_res = np.sum((true_thickness - pred_thickness) ** 2)
    ss_tot = np.sum((true_thickness - np.mean(true_thickness)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    metrics = {
        '覆冰厚度_MAE': mae,
        '覆冰厚度_RMSE': rmse,
        '覆冰厚度_R2': r2,
        '覆冰比值_MAE': mae_ratio,
        '覆冰比值_RMSE': rmse_ratio,
        **accuracies,
        '总预测数': len(preds),
    }
    return metrics


def compute_window_metrics(preds, labels, time_steps):
    """按预测时间窗口分析精度"""
    pred_thickness = preds[:, 0]
    true_thickness = labels[:, 0]
    errors = np.abs(pred_thickness - true_thickness)
    
    # 将时间步转为小时（每6分钟1条记录）
    hours = time_steps * 6 / 60.0
    
    windows = [(0, 6), (6, 12), (12, 24), (24, 48)]
    window_metrics = {}
    
    for w_start, w_end in windows:
        mask = (hours >= w_start) & (hours < w_end)
        if mask.sum() > 0:
            # Convert numpy scalar to Python float for JSON serialization.
            w_mae = float(np.mean(errors[mask]))
            w_acc = float(np.mean(errors[mask] < 0.2))
            window_metrics[f'{w_start}-{w_end}h'] = {
                'MAE': w_mae, 'Acc@0.2': w_acc, 'count': int(mask.sum())
            }
    
    return window_metrics


def main():
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    # 1. 加载模型
    print("加载模型...")
    model = Seq2ABTransformer(condition_length=cfg['cdt_length']).to(cfg['device'])
    
    state_dict = torch.load(cfg['weights_path'], map_location=cfg['device'])
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    print(f"  模型加载完成: {cfg['weights_path']}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 2. 加载数据
    print("加载数据集...")
    dataset = FubingDataset(batch_size=cfg['batch_size'])
    print(f"  数据集大小: {dataset.total_length} 样本, {dataset.num_types} 个终端")
    
    # 3. 评估
    preds, labels, errors, time_steps = evaluate_model(model, dataset, cfg)
    
    # 4. 计算指标
    metrics = compute_metrics(preds, labels, errors)
    window_metrics = compute_window_metrics(preds, labels, time_steps)
    
    # 5. 打印结果
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n按预测时间窗口分析:")
    for window, wm in window_metrics.items():
        print(f"  {window}: MAE={wm['MAE']:.4f}, Acc@0.2={wm['Acc@0.2']:.4f} (n={wm['count']})")
    
    # 6. 保存结果
    # 保存预测数据（供绘图脚本使用）
    np.savez(os.path.join(cfg['output_dir'], 'predictions.npz'),
             preds=preds, labels=labels, errors=errors, time_steps=time_steps)
    
    # 保存评估报告
    report = {
        'metrics': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()},
        'window_metrics': window_metrics,
        'model_params': {'total': total_params, 'trainable': trainable_params},
        'config': {k: str(v) for k, v in cfg.items()},
    }
    with open(os.path.join(cfg['output_dir'], 'eval_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 人类可读报告
    with open(os.path.join(cfg['output_dir'], 'eval_report.txt'), 'w', encoding='utf-8') as f:
        f.write("Seq2ABTransformer 覆冰厚度预测 — 评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)\n")
        f.write(f"测试样本数: {metrics['总预测数']}\n\n")
        f.write("--- 覆冰厚度预测指标 ---\n")
        f.write(f"MAE:  {metrics['覆冰厚度_MAE']:.4f}\n")
        f.write(f"RMSE: {metrics['覆冰厚度_RMSE']:.4f}\n")
        f.write(f"R²:   {metrics['覆冰厚度_R2']:.4f}\n\n")
        f.write("--- 多阈值准确率 ---\n")
        for thresh in [0.1, 0.2, 0.3, 0.5, 1.0]:
            f.write(f"Acc@{thresh}: {metrics[f'Acc@{thresh}']:.4f}\n")
        f.write("\n--- 不同预测窗口精度 ---\n")
        for window, wm in window_metrics.items():
            f.write(f"{window}: MAE={wm['MAE']:.4f}, Acc@0.2={wm['Acc@0.2']:.4f}\n")
    
    print(f"\n[OK] 结果已保存到: {cfg['output_dir']}")
    print("  - predictions.npz (预测数据，供 plot_experiments.py 使用)")
    print("  - eval_report.json (机器可读报告)")
    print("  - eval_report.txt (人类可读报告，可引用到研究报告)")


if __name__ == '__main__':
    main()
