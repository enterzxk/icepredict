"""
实验图表生成脚本 — 基于evaluate.py导出的predictions.npz生成所有实验图表
供研究报告第5章和第6章使用

生成图表：
  1. 预测值 vs 真实值散点图
  2. 预测误差分布直方图
  3. 多阈值准确率条形图
  4. 不同预测窗口精度对比
  5. 预测值 vs 真实值时序曲线（选取连续片段）

使用方法（在evaluate.py之后运行）：
  python experiments/plot_experiments.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import json

# ========== 中文字体配置（兼容Windows/Linux）==========
import matplotlib.font_manager as fm
import platform

def setup_chinese_font():
    """自动检测并设置中文字体"""
    system = platform.system()
    if system == 'Windows':
        candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    else:
        candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.sans-serif'] = [font]
            return font
    # 最终备选：直接用字体文件路径（Windows微软雅黑）
    win_font = r'C:\Windows\Fonts\msyh.ttc'
    if os.path.exists(win_font):
        fm.fontManager.addfont(win_font)
        prop = fm.FontProperties(fname=win_font)
        matplotlib.rcParams['font.sans-serif'] = [prop.get_name()]
        return prop.get_name()
    return 'DejaVu Sans'

setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def load_results():
    """加载评估结果"""
    data = np.load(os.path.join(RESULTS_DIR, 'predictions.npz'))
    report_path = os.path.join(RESULTS_DIR, 'eval_report.json')
    report = {}
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
    return data['preds'], data['labels'], data['errors'], data['time_steps'], report


def plot_pred_vs_true(preds, labels):
    """图1: 预测值 vs 真实值散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 覆冰厚度
    p, l = preds[:, 0], labels[:, 0]
    ax1.scatter(l, p, s=3, alpha=0.3, c='steelblue')
    lim = max(max(abs(p.max()), abs(l.max())), 1)
    ax1.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5, label='理想预测线')
    ax1.set_xlabel('真实覆冰厚度', fontsize=11)
    ax1.set_ylabel('预测覆冰厚度', fontsize=11)
    ax1.set_title('覆冰厚度：预测值 vs 真实值', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 覆冰比值
    p2, l2 = preds[:, 1], labels[:, 1]
    ax2.scatter(l2, p2, s=3, alpha=0.3, c='darkorange')
    lim2 = max(max(abs(p2.max()), abs(l2.max())), 1)
    ax2.plot([-lim2, lim2], [-lim2, lim2], 'r--', linewidth=1.5, label='理想预测线')
    ax2.set_xlabel('真实覆冰比值', fontsize=11)
    ax2.set_ylabel('预测覆冰比值', fontsize=11)
    ax2.set_title('覆冰比值：预测值 vs 真实值', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, 'exp_fig1_pred_vs_true')


def plot_error_distribution(errors):
    """图2: 预测误差分布"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=80, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=1.5, 
               label=f'MAE = {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=1.5,
               label=f'中位数 = {np.median(errors):.4f}')
    ax.set_xlabel('预测绝对误差', fontsize=11)
    ax.set_ylabel('概率密度', fontsize=11)
    ax.set_title('覆冰厚度预测误差分布', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'exp_fig2_error_distribution')


def plot_accuracy_thresholds(preds, labels):
    """图3: 多阈值准确率条形图"""
    pred_thickness = preds[:, 0]
    true_thickness = labels[:, 0]
    abs_errors = np.abs(pred_thickness - true_thickness)
    
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    accs = [np.mean(abs_errors < t) * 100 for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(thresholds)))
    bars = ax.bar([f'{t}' for t in thresholds], accs, color=colors, edgecolor='white', width=0.6)
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('误差阈值', fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_title('不同误差阈值下的预测准确率', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'exp_fig3_accuracy_thresholds')


def plot_window_analysis(preds, labels, time_steps):
    """图4: 不同预测时间窗口的精度对比"""
    pred_thickness = preds[:, 0]
    true_thickness = labels[:, 0]
    errors = np.abs(pred_thickness - true_thickness)
    hours = time_steps * 6 / 60.0  # 转为小时
    
    windows = [(0, 3), (3, 6), (6, 12), (12, 24), (24, 48)]
    window_labels = []
    window_maes = []
    window_accs = []
    
    for w_start, w_end in windows:
        mask = (hours >= w_start) & (hours < w_end)
        if mask.sum() > 0:
            window_labels.append(f'{w_start}-{w_end}h')
            window_maes.append(np.mean(errors[mask]))
            window_accs.append(np.mean(errors[mask] < 0.2) * 100)
    
    if not window_labels:
        print("  ⚠ 无法生成窗口分析图（数据不足）")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = range(len(window_labels))
    ax1.bar(x, window_maes, color='steelblue', edgecolor='white', width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(window_labels)
    ax1.set_xlabel('预测时间窗口', fontsize=11)
    ax1.set_ylabel('MAE', fontsize=11)
    ax1.set_title('各窗口MAE对比', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(x, window_accs, color='darkorange', edgecolor='white', width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(window_labels)
    ax2.set_xlabel('预测时间窗口', fontsize=11)
    ax2.set_ylabel('准确率 (%)', fontsize=11)
    ax2.set_title('各窗口Acc@0.2对比', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, 'exp_fig4_window_analysis')


def plot_prediction_curve(preds, labels):
    """图5: 预测曲线 vs 真实曲线（取一段连续样本展示）"""
    n_show = min(200, len(preds))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    x = range(n_show)
    # 覆冰厚度
    ax1.plot(x, labels[:n_show, 0], 'b-', linewidth=1, label='真实值', alpha=0.8)
    ax1.plot(x, preds[:n_show, 0], 'r-', linewidth=1, label='预测值', alpha=0.8)
    ax1.fill_between(x, labels[:n_show, 0], preds[:n_show, 0], alpha=0.15, color='red')
    ax1.set_ylabel('覆冰厚度', fontsize=11)
    ax1.set_title('覆冰厚度预测结果对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # 覆冰比值
    ax2.plot(x, labels[:n_show, 1], 'b-', linewidth=1, label='真实值', alpha=0.8)
    ax2.plot(x, preds[:n_show, 1], 'r-', linewidth=1, label='预测值', alpha=0.8)
    ax2.fill_between(x, labels[:n_show, 1], preds[:n_show, 1], alpha=0.15, color='red')
    ax2.set_xlabel('样本序号', fontsize=11)
    ax2.set_ylabel('覆冰比值', fontsize=11)
    ax2.set_title('覆冰比值预测结果对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, 'exp_fig5_prediction_curve')


def generate_metrics_table(report):
    """生成可直接粘贴到报告的指标表格（Markdown格式）"""
    if not report or 'metrics' not in report:
        return
    
    m = report['metrics']
    
    table = """
| 指标 | 数值 |
|------|------|
| MAE (覆冰厚度) | {:.4f} |
| RMSE (覆冰厚度) | {:.4f} |
| R² (覆冰厚度) | {:.4f} |
| MAE (覆冰比值) | {:.4f} |
| RMSE (覆冰比值) | {:.4f} |
| Acc@0.1 | {:.1f}% |
| Acc@0.2 | {:.1f}% |
| Acc@0.3 | {:.1f}% |
| Acc@0.5 | {:.1f}% |
| Acc@1.0 | {:.1f}% |
""".format(
        m['覆冰厚度_MAE'], m['覆冰厚度_RMSE'], m['覆冰厚度_R2'],
        m['覆冰比值_MAE'], m['覆冰比值_RMSE'],
        m['Acc@0.1']*100, m['Acc@0.2']*100, m['Acc@0.3']*100, m['Acc@0.5']*100, m['Acc@1.0']*100
    )
    
    path = os.path.join(OUTPUT_DIR, 'metrics_table.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(table)
    print(f"  [OK] 指标表格已保存: {path}")


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 已保存: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pred_path = os.path.join(RESULTS_DIR, 'predictions.npz')
    if not os.path.exists(pred_path):
        print("❌ 未找到 predictions.npz，请先运行 evaluate.py")
        print(f"   期望路径: {pred_path}")
        return
    
    preds, labels, errors, time_steps, report = load_results()
    print(f"加载预测结果: {len(preds)} 条记录\n")
    
    print("正在生成实验图表...")
    plot_pred_vs_true(preds, labels)
    plot_error_distribution(errors)
    plot_accuracy_thresholds(preds, labels)
    plot_window_analysis(preds, labels, time_steps)
    plot_prediction_curve(preds, labels)
    generate_metrics_table(report)
    
    print(f"\n[OK] 所有实验图表已生成到: {OUTPUT_DIR}")
    print("这些图表可直接用于研究报告第6.4节（覆冰厚度预测实验）")


if __name__ == '__main__':
    main()
