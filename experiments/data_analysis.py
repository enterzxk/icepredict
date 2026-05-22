"""
数据探索性分析（EDA） — 4C大赛研究报告用图
生成图表：
  1. 各终端数据量分布柱状图
  2. 覆冰厚度时序变化趋势图  
  3. 温度-湿度-覆冰厚度相关性热力图
  4. 覆冰厚度分布直方图
  5. 覆冰/非覆冰样本比例饼图
  6. 温度与覆冰厚度散点图

使用方法：
  python experiments/data_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings
warnings.filterwarnings('ignore')

# ========== 中文显示配置 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

# ========== 配置 ==========
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data', 'merged_data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')

def load_data():
    """加载并预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # 转换数值列
    for col in ['覆冰厚度', '覆冰比值', '温度', '湿度']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['覆冰厚度', '温度', '湿度', '时间'])
    print(f"数据加载完成: {len(df)} 条记录, {df['终端编号'].nunique()} 个终端")
    return df


def plot_terminal_distribution(df):
    """图1: 各终端数据量分布"""
    fig, ax = plt.subplots(figsize=(12, 5))
    counts = df['终端编号'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('终端编号', fontsize=12)
    ax.set_ylabel('数据条数', fontsize=12)
    ax.set_title('各监测终端数据量分布', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # 添加总数标注
    total = counts.sum()
    ax.text(0.98, 0.95, f'总计: {total:,} 条\n终端数: {len(counts)} 个', 
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    save_fig(fig, 'fig1_terminal_distribution')


def plot_ice_thickness_timeseries(df):
    """图2: 典型终端覆冰厚度时序变化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    # 选4个有代表性的终端
    terminals = df['终端编号'].value_counts().head(4).index.tolist()
    
    for idx, (ax, terminal) in enumerate(zip(axes.flatten(), terminals)):
        tdf = df[df['终端编号'] == terminal].sort_values('时间')
        ax.plot(tdf['时间'], tdf['覆冰厚度'], linewidth=0.5, color=plt.cm.tab10(idx), alpha=0.8)
        ax.set_title(f'终端 {terminal}', fontsize=11, fontweight='bold')
        ax.set_ylabel('覆冰厚度 (mm)', fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        ax.grid(alpha=0.3)
    
    fig.suptitle('典型终端覆冰厚度时序变化趋势', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig2_ice_thickness_timeseries')


def plot_correlation_heatmap(df):
    """图3: 温度-湿度-覆冰相关性热力图"""
    fig, ax = plt.subplots(figsize=(7, 6))
    cols = ['覆冰厚度', '覆冰比值', '温度', '湿度']
    corr = df[cols].corr()
    
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticklabels(cols, fontsize=11)
    
    # 添加数值标注
    for i in range(len(cols)):
        for j in range(len(cols)):
            color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr.iloc[i, j]:.3f}', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson相关系数')
    ax.set_title('传感器特征相关性矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'fig3_correlation_heatmap')


def plot_ice_distribution(df):
    """图4: 覆冰厚度分布直方图 + 覆冰/非覆冰比例"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图
    ice_data = df['覆冰厚度']
    ax1.hist(ice_data[ice_data > 0], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('覆冰厚度 (mm)', fontsize=11)
    ax1.set_ylabel('频次', fontsize=11)
    ax1.set_title('覆冰厚度分布（非零值）', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.axvline(ice_data[ice_data > 0].mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'均值: {ice_data[ice_data > 0].mean():.2f}')
    ax1.legend(fontsize=10)
    
    # 饼图
    zero_count = (ice_data == 0).sum()
    nonzero_count = (ice_data > 0).sum()
    labels = ['无覆冰 (厚度=0)', '有覆冰 (厚度>0)']
    sizes = [zero_count, nonzero_count]
    colors_pie = ['#66b3ff', '#ff6666']
    explode = (0, 0.05)
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    for t in autotexts:
        t.set_fontweight('bold')
    ax2.set_title('覆冰/非覆冰样本比例', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'fig4_ice_distribution')


def plot_temp_vs_ice(df):
    """图5: 温度-覆冰厚度散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sample = df.sample(min(20000, len(df)), random_state=42)
    
    # 温度 vs 覆冰厚度
    ax1.scatter(sample['温度'], sample['覆冰厚度'], s=1, alpha=0.3, c='steelblue')
    ax1.set_xlabel('温度 (°C)', fontsize=11)
    ax1.set_ylabel('覆冰厚度 (mm)', fontsize=11)
    ax1.set_title('温度与覆冰厚度关系', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 湿度 vs 覆冰厚度
    ax2.scatter(sample['湿度'], sample['覆冰厚度'], s=1, alpha=0.3, c='darkorange')
    ax2.set_xlabel('湿度 (%)', fontsize=11)
    ax2.set_ylabel('覆冰厚度 (mm)', fontsize=11)
    ax2.set_title('湿度与覆冰厚度关系', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, 'fig5_temp_humidity_vs_ice')


def plot_hourly_pattern(df):
    """图6: 覆冰厚度的小时分布模式"""
    fig, ax = plt.subplots(figsize=(10, 5))
    df_copy = df.copy()
    df_copy['hour'] = df_copy['时间'].dt.hour
    hourly = df_copy.groupby('hour')['覆冰厚度'].agg(['mean', 'std']).reset_index()
    
    ax.bar(hourly['hour'], hourly['mean'], color='steelblue', alpha=0.7, edgecolor='white')
    ax.errorbar(hourly['hour'], hourly['mean'], yerr=hourly['std'], fmt='none', 
                ecolor='gray', capsize=3, alpha=0.5)
    ax.set_xlabel('小时 (24h)', fontsize=11)
    ax.set_ylabel('平均覆冰厚度 (mm)', fontsize=11)
    ax.set_title('覆冰厚度的日内变化模式', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'fig6_hourly_pattern')


def generate_data_summary(df):
    """生成数据集统计摘要表格（直接用于报告）"""
    summary = {
        '数据集总量': f'{len(df):,} 条',
        '监测终端数': f'{df["终端编号"].nunique()} 个',
        '时间跨度': f'{df["时间"].min().strftime("%Y-%m-%d")} ~ {df["时间"].max().strftime("%Y-%m-%d")}',
        '采集频率': '约6分钟/条',
        '覆冰厚度范围': f'{df["覆冰厚度"].min():.2f} ~ {df["覆冰厚度"].max():.2f} mm',
        '覆冰厚度均值': f'{df["覆冰厚度"].mean():.4f} mm',
        '温度范围': f'{df["温度"].min():.1f} ~ {df["温度"].max():.1f} °C',
        '湿度范围': f'{df["湿度"].min():.1f} ~ {df["湿度"].max():.1f} %',
        '有覆冰样本占比': f'{(df["覆冰厚度"] > 0).mean()*100:.1f}%',
    }
    
    print("\n" + "="*50)
    print("数据集统计摘要（可直接引用到报告中）")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("="*50)
    
    # 保存为txt
    with open(os.path.join(OUTPUT_DIR, 'data_summary.txt'), 'w', encoding='utf-8') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


def save_fig(fig, name):
    """保存图表"""
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 已保存: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    
    print("\n正在生成图表...")
    plot_terminal_distribution(df)
    plot_ice_thickness_timeseries(df)
    plot_correlation_heatmap(df)
    plot_ice_distribution(df)
    plot_temp_vs_ice(df)
    plot_hourly_pattern(df)
    generate_data_summary(df)
    
    print(f"\n[OK] 所有EDA图表已生成到: {OUTPUT_DIR}")
    print("这些图表可直接用于研究报告第2章（数据来源与分析）和第6.2节（数据探索性分析）")


if __name__ == '__main__':
    main()
