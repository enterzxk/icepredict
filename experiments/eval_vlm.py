"""
VLM覆冰图像识别评估脚本
对比人工标注 vs VLM自动识别结果，计算P/R/F1，绘制混淆矩阵

使用前准备：
  1. 先运行 snow_ice_detector_plus_v2.py 得到 snow_ice_result.json
  2. 人工标注：创建 human_labels.json，格式如下：
     [
       {"image_path": "xxx.jpg", "label": "yes"},
       {"image_path": "xxx.jpg", "label": "no"},
       ...
     ]
     label值: "yes"=有覆冰/积雪, "no"=无覆冰

使用方法：
  python eval_vlm.py --result snow_ice_result.json --labels human_labels.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import argparse
from collections import Counter

# ========== 中文配置 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'eval_figures')


def load_data(result_path, labels_path):
    """加载VLM识别结果和人工标注"""
    with open(result_path, 'r', encoding='utf-8') as f:
        vlm_results = json.load(f)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        human_labels = json.load(f)
    
    # 构建路径→标签映射
    vlm_map = {}
    for item in vlm_results:
        path = item.get('image_path', '')
        fname = os.path.basename(path)
        vlm_map[fname] = item.get('result', 'unknow')
    
    human_map = {}
    for item in human_labels:
        path = item.get('image_path', '')
        fname = os.path.basename(path)
        human_map[fname] = item.get('label', 'unknow')
    
    # 找到共同的图片
    common = set(vlm_map.keys()) & set(human_map.keys())
    print(f"VLM识别结果: {len(vlm_map)} 张")
    print(f"人工标注: {len(human_map)} 张")
    print(f"共同匹配: {len(common)} 张")
    
    y_true = []
    y_pred = []
    for fname in common:
        y_true.append(human_map[fname])
        y_pred.append(vlm_map[fname])
    
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    """计算 Precision, Recall, F1"""
    # 二分类：yes=正类, no=负类
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 'yes' and p == 'yes')
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 'no' and p == 'yes')
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 'yes' and p != 'yes')
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 'no' and p != 'yes')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    metrics = {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy,
        '总样本数': len(y_true),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, metrics):
    """绘制混淆矩阵热力图"""
    cm = np.array([
        [metrics['TN'], metrics['FP']],
        [metrics['FN'], metrics['TP']]
    ])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['无覆冰 (No)', '有覆冰 (Yes)'], fontsize=12)
    ax.set_yticklabels(['无覆冰 (No)', '有覆冰 (Yes)'], fontsize=12)
    ax.set_xlabel('VLM预测结果', fontsize=13)
    ax.set_ylabel('人工标注（真实值）', fontsize=13)
    
    # 添加数值
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   fontsize=20, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('覆冰图像识别混淆矩阵', fontsize=14, fontweight='bold')
    
    # 添加指标文本
    text = f'Precision: {metrics["Precision"]:.3f}\nRecall: {metrics["Recall"]:.3f}\nF1: {metrics["F1-Score"]:.3f}\nAccuracy: {metrics["Accuracy"]:.3f}'
    ax.text(1.35, 0.5, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_fig(fig, 'vlm_confusion_matrix')


def plot_metrics_bar(metrics):
    """绘制P/R/F1条形图"""
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [metrics[n] * 100 for n in names]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    
    bars = ax.bar(names, values, color=colors, edgecolor='white', width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('百分比 (%)', fontsize=12)
    ax.set_title('VLM覆冰图像识别性能指标', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'vlm_metrics_bar')


def generate_report(metrics):
    """生成评估报告"""
    report = f"""
VLM覆冰图像识别 — 评估报告
{'='*50}

模型: 通义千问 qwen3.5-flash (VLM)
推理方式: 零样本 (Zero-shot)
评估样本数: {metrics['总样本数']}

--- 分类性能 ---
Precision (精确率): {metrics['Precision']:.4f}  ({metrics['Precision']*100:.1f}%)
Recall (召回率):    {metrics['Recall']:.4f}  ({metrics['Recall']*100:.1f}%)
F1-Score:           {metrics['F1-Score']:.4f}  ({metrics['F1-Score']*100:.1f}%)
Accuracy (准确率):  {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.1f}%)

--- 混淆矩阵 ---
              预测No    预测Yes
真实No         {metrics['TN']:4d}       {metrics['FP']:4d}
真实Yes        {metrics['FN']:4d}       {metrics['TP']:4d}

--- 用于报告的Markdown表格 ---

| 指标 | 数值 |
|------|------|
| Precision | {metrics['Precision']*100:.1f}% |
| Recall | {metrics['Recall']*100:.1f}% |
| F1-Score | {metrics['F1-Score']*100:.1f}% |
| Accuracy | {metrics['Accuracy']*100:.1f}% |
"""
    
    path = os.path.join(OUTPUT_DIR, 'vlm_eval_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [OK] 评估报告: {path}")
    print(report)


def create_label_template(image_dir, output_path):
    """辅助工具：生成人工标注模板文件"""
    from pathlib import Path
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    image_files = []
    for path in Path(image_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(str(path.resolve()))
    
    image_files.sort()
    template = [{"image_path": p, "label": ""} for p in image_files]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 标注模板已生成: {output_path}")
    print(f"   包含 {len(template)} 张图片，请在 label 字段填入 'yes' 或 'no'")


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description='VLM覆冰识别评估')
    parser.add_argument('--result', type=str, default='snow_ice_result.json', help='VLM识别结果文件')
    parser.add_argument('--labels', type=str, default='human_labels.json', help='人工标注文件')
    parser.add_argument('--gen-template', type=str, default='', help='生成标注模板，传入图片目录路径')
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 如果需要生成标注模板
    if args.gen_template:
        create_label_template(args.gen_template, 'human_labels.json')
        return
    
    # 检查文件
    if not os.path.exists(args.result):
        print(f"❌ VLM结果文件不存在: {args.result}")
        print("   请先运行 snow_ice_detector_plus_v2.py")
        return
    if not os.path.exists(args.labels):
        print(f"❌ 人工标注文件不存在: {args.labels}")
        print(f"   请先创建标注文件，或运行:")
        print(f"   python eval_vlm.py --gen-template 部分样本")
        return
    
    y_true, y_pred = load_data(args.result, args.labels)
    
    if len(y_true) == 0:
        print("❌ 无匹配数据")
        return
    
    metrics = compute_metrics(y_true, y_pred)
    
    print("\n正在生成图表...")
    plot_confusion_matrix(y_true, y_pred, metrics)
    plot_metrics_bar(metrics)
    generate_report(metrics)
    
    print(f"\n[OK] 所有图表已生成到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
