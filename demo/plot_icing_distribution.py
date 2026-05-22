import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 全局字体与格式设置
# 解决中文乱码和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # 兼容不同系统的中文字体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 设置图表全局分辨率与尺寸
plt.rcParams['figure.dpi'] = 300
fig = plt.figure(figsize=(14, 6))

# 2. 数据读取与预处理
csv_path = '../data/data/merged_data.csv'  # 请确保运行路径正确，根据您的描述，这是相对路径

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    # 兼容绝对路径运行
    df = pd.read_csv(r'c:\Users\13412\Desktop\2026023169-作品主文件夹\2026023169-02素材与源码\data\data\merged_data.csv')

# 列名映射 (根据实际CSV列名调整)
temp_col = '温度'
humid_col = '湿度'
ice_col = '覆冰厚度'

# 过滤无效数据 (覆冰厚度 > 0)
df_valid = df[df[ice_col] > 0].copy()

# 抗锯齿/防重叠：随机抽取 10000 条样本
if len(df_valid) > 10000:
    df_sample = df_valid.sample(n=10000, random_state=42)
else:
    df_sample = df_valid

x = df_sample[temp_col]
y = df_sample[humid_col]
z = df_sample[ice_col]

# 3. 绘制左图：二维颜色映射散点图
ax1 = fig.add_subplot(1, 2, 1)
# 使用极低 alpha 防重叠，coolwarm 渐变色
scatter1 = ax1.scatter(x, y, c=z, cmap='coolwarm', alpha=0.5, edgecolors='none', s=15)
ax1.set_xlabel('温度 (℃)', fontsize=12)
ax1.set_ylabel('湿度 (%)', fontsize=12)
ax1.set_title('温度与湿度的二维联合分布及覆冰厚度映射', fontsize=14, pad=15)
ax1.grid(True, linestyle='--', alpha=0.3)

# 添加 Colorbar
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('覆冰厚度 (mm)', fontsize=11)

# 4. 绘制右图：三维散点图
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scatter2 = ax2.scatter(x, y, z, c=z, cmap='coolwarm', alpha=0.6, edgecolors='none', s=10)
ax2.set_xlabel('温度 (℃)', fontsize=11)
ax2.set_ylabel('湿度 (%)', fontsize=11)
ax2.set_zlabel('覆冰厚度 (mm)', fontsize=11)
ax2.set_title('温湿双重驱动下的覆冰厚度三维空间分布', fontsize=14, pad=15)

# 调整优雅的初始观察视角
ax2.view_init(elev=20, azim=45)

# 5. 调整布局与保存
plt.tight_layout()
plt.savefig('icing_joint_distribution.png', bbox_inches='tight')
print("图表已成功保存为 icing_joint_distribution.png")

# 6. 显示图表
plt.show()
