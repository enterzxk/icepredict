# 项目拆解与运行指南

## 1. 项目定位

这是一个输电线路覆冰智能监测与预警项目，项目名在前端中展示为 IceGuard AI。它把两类信息结合起来：

- 传感器时序数据：覆冰厚度、覆冰比值、温度、湿度、时间、终端编号。
- 监控图像数据：图片是否存在覆冰、积雪或霜冻。

核心思路是：用 `Seq2ABTransformer` 根据历史传感器序列和未来气象条件预测未来覆冰厚度/覆冰比值，再用图像识别结果做交叉验证，最后输出分级预警。

图像识别支持两种模式：
- **VLM模式**：使用阿里云DashScope VLM API进行零样本识别（需要API Key）
- **本地模式**：使用本地训练的ResNet50多标签分类模型（推荐）

## 2. 顶层目录

```text
.
├─ data/
│  ├─ data/merged_data.csv              # 合并后的传感器时序数据，287888 行
│  ├─ imagine/                          # 原始/候选图像（644张）
│  ├─ labels/                           # 多标签标注结果
│  └─ dataset/                          # 划分后的数据集（train/val/test）
├─ data_shards/                         # 已切片的训练/测试 CSV 样本
├─ demo/                                # 静态前端展示页
│  ├─ index.html
│  ├─ style.css
│  ├─ app.js
│  ├─ data.json                         # demo 动态图像与预测曲线数据
│  └─ assets/
├─ experiments/                         # 实验、评估、画图脚本
│  ├─ data_analysis.py                  # EDA 数据分析图表
│  ├─ evaluate.py                       # 加载权重并重新评估模型
│  ├─ plot_experiments.py               # 基于已有 predictions.npz 绘制实验图
│  ├─ eval_vlm.py                       # VLM 图像识别评估
│  ├─ evaluate_ice_classifier.py        # 本地图像分类器评估
│  ├─ results/                          # 已有预测结果与评估报告
│  └─ figures/                          # 生成好的图表
├─ ice_monitor/                         # 推理与预警封装
│  ├─ predictor.py                      # 载入模型权重并预测
│  ├─ alert.py                          # 融合预警规则（支持VLM和本地模型）
│  ├─ local_detector.py                 # 本地图像检测器
│  └─ vlm/detector.py                   # DashScope/OpenAI-compatible VLM 调用
├─ src/
│  ├─ model.py                          # Seq2ABTransformer
│  ├─ model_utils.py                    # 位置编码、DropPath 等工具
│  ├─ dataset.py                        # merged_data.csv -> 变长历史/未来序列
│  ├─ dataset_new.py                    # data_shards 版本的数据集封装
│  ├─ process_csv_files.py              # CSV 分片/拆分工具
│  ├─ ice_classifier.py                 # 覆冰图像分类器模型（ResNet50）
│  ├─ ice_dataset.py                    # 覆冰图像数据集类
│  └─ data_augmentation.py              # 数据增强模块
├─ tools/                               # 工具脚本
│  ├─ auto_label.py                     # VLM自动标注工具
│  └─ split_dataset.py                  # 数据集划分工具
├─ train_ice_classifier.py              # 本地图像分类器训练脚本
├─ evaluate_ice_classifier.py           # 本地图像分类器评估脚本
├─ weights/
│  ├─ 20251205/best.ckpt                # 时序预测模型权重
│  └─ ice_classifier/                   # 图像分类器权重
│      ├─ best_stage1.pth               # 第一阶段最佳模型
│      └─ best_stage2.pth               # 第二阶段最佳模型（推荐）
└─ requirements.txt
```

## 3. 数据流

### 3.1 原始合并数据

`data/data/merged_data.csv` 有 287888 行、6 列：

```text
覆冰厚度, 覆冰比值, 温度, 湿度, 时间, 终端编号
```

当前数据统计：

- 终端数：24 个。
- 时间范围：2024-01-01 00:00:22 到 2024-12-31 23:58:09。
- 覆冰厚度范围：0.00 到 26.29 mm。
- 覆冰厚度均值：0.8803 mm。
- 有覆冰样本占比：约 23.1%。

### 3.2 `src/dataset.py`

`FubingDataset` 按终端编号分组，为每个当前时刻构造：

- `batch_pre`：当前时刻之前约 6 小时的历史序列，并拼上当前点。
- `batch_post`：当前时刻之后最多约 2 天的未来序列。

每条序列的单步特征是 5 维：

```text
[覆冰厚度, 覆冰比值, 温度, 湿度, 相对时间]
```

处理细节：

- 时间转成相对天数。
- 湿度除以 100。
- 评估时会从未来序列中抽取多个预测时间点。
- 未来条件只取 `[温度, 湿度, 相对时间]`，展平成长度不超过 1440 的向量，不足补 0。

第一次运行 `experiments/evaluate.py` 时，如果没有 `data/data/data_dict.pkl`，会先构建缓存，可能耗时较久。

## 4. 模型结构

核心模型在 `src/model.py`：`Seq2ABTransformer`。

输入：

- `hist_x`: 历史序列，形状 `(batch, seq_len, 5)`。
- `curr_cdt`: 未来条件展平向量，训练权重对应长度是 `1440`。

输出：

```text
[预测覆冰厚度, 预测覆冰比值]
```

结构：

1. `MLP1`: 单步 5 维历史特征映射到 64 维。
2. `CLS Token`: 拼到序列开头，用来聚合历史序列全局信息。
3. `AttentionLayer x 3`: 3 层自注意力，8 个头。
4. `MLP2`: 未来条件向量 `1440 -> 32 -> 64`。
5. `MLP3`: 拼接历史 CLS 特征和未来条件特征，`128 -> 64 -> 32 -> 2`。

当前已做过兼容性修复：如果本机没有 `xformers`，模型会自动使用 PyTorch 原生 `scaled_dot_product_attention`，所以 CPU 环境也能跑。

## 5. 运行环境

已在当前机器验证：

- Python 3.13.9
- torch 2.9.1+cpu
- numpy 2.1.3
- pandas 3.0.2
- matplotlib 3.10.8
- einops 0.8.2
- kornia 0.8.2

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

`openai` 只在使用 `ice_monitor/vlm/detector.py` 时需要。VLM 接口走 DashScope 的 OpenAI-compatible endpoint，还需要设置：

```powershell
$env:DASHSCOPE_API_KEY="你的 DashScope API Key"
```

## 6. 推荐运行顺序

所有命令都在项目根目录执行：

```powershell
cd "C:\Users\43106\Desktop\2026023169-作品主文件夹\2026023169-02素材与源码"
```

### 6.1 编译检查

```powershell
python -m py_compile src\model.py experiments\data_analysis.py experiments\plot_experiments.py experiments\evaluate.py ice_monitor\predictor.py ice_monitor\alert.py ice_monitor\vlm\detector.py
```

### 6.2 生成数据分析图

```powershell
python experiments\data_analysis.py
```

输出到：

```text
experiments/figures/
```

### 6.3 生成模型实验图

这个脚本使用已有的 `experiments/results/predictions.npz`，速度较快：

```powershell
python experiments\plot_experiments.py
```

输出包括：

- `exp_fig1_pred_vs_true.png`
- `exp_fig2_error_distribution.png`
- `exp_fig3_accuracy_thresholds.png`
- `exp_fig4_window_analysis.png`
- `exp_fig5_prediction_curve.png`
- `metrics_table.md`

### 6.4 重新评估模型

```powershell
python experiments\evaluate.py
```

注意：

- 会加载 `weights/20251205/best.ckpt`。
- 会读取 `data/data/merged_data.csv`。
- 会生成/覆盖 `experiments/results/predictions.npz`、`eval_report.json`、`eval_report.txt`。
- 首次构建数据缓存可能较慢。
- 有 CUDA 时自动用 `cuda:0`，否则用 CPU。

### 6.5 跑最小模型推理

```powershell
python -c "import numpy as np; from ice_monitor.predictor import IceThicknessPredictor; p=IceThicknessPredictor(device='cpu'); h=np.zeros((6,5),dtype=np.float32); f=np.zeros((3,3),dtype=np.float32); print(p.predict(h,f))"
```

返回格式：

```text
{'ice_thickness': ..., 'ice_ratio': ..., 'pred_steps': 3}
```

### 6.6 打开前端 demo

```powershell
cd demo
python -m http.server 8000
```

然后打开：

```text
http://localhost:8000/
```

demo 是静态前端展示页，读取 `demo/data.json`，不会实时调用 Python 模型。`demo/data.json` 里有：

- `images`: 123 张图像及是否覆冰标签。
- `predictions`: 48 个预测曲线点。

### 6.7 本地图像分类器（新功能）

本项目新增了本地图像分类器，使用ResNet50进行多标签分类，可识别：覆冰、雪、积雪、霜冻。

#### 6.7.1 数据准备

首先使用VLM对图像进行自动标注：

```powershell
python tools/auto_label.py --image-dir data/imagine --output multi_label_results.json
```

然后转换为训练格式：

```powershell
python tools/auto_label.py --convert --convert-input multi_label_results.json --convert-output data/labels
```

最后划分数据集：

```powershell
python tools/split_dataset.py --label-file data/labels/training_labels.json --output-dir data/dataset
```

#### 6.7.2 训练模型

两阶段训练：

```powershell
python train_ice_classifier.py --data-dir data/dataset --use-focal-loss --use-mixup
```

训练参数说明：
- `--epochs-stage1 10`：第一阶段训练轮数（冻结backbone）
- `--epochs-stage2 30`：第二阶段训练轮数（全参数微调）
- `--batch-size 32`：批次大小
- `--lr-stage1 1e-3`：第一阶段学习率
- `--lr-stage2 1e-4`：第二阶段学习率

训练结果保存在 `weights/ice_classifier/` 目录。

#### 6.7.3 评估模型

```powershell
python evaluate_ice_classifier.py --checkpoint weights/ice_classifier/best_stage2.pth --data-dir data/dataset
```

评估结果保存在 `experiments/eval_figures/` 目录，包括：
- 混淆矩阵
- ROC曲线
- PR曲线
- 指标条形图
- 评估报告

#### 6.7.4 使用本地图型推理

```python
from ice_monitor.local_detector import LocalIceDetector

detector = LocalIceDetector()
result = detector.detect("path/to/image.jpg")
print(result)
```

输出格式：
```python
{
    'image_path': 'path/to/image.jpg',
    'label': '覆冰+雪',
    'confidence': 0.85,
    'details': {'覆冰': 0.9, '雪': 0.8, '积雪': 0.1, '霜冻': 0.05},
    'labels': ['覆冰', '雪']
}
```

#### 6.7.5 在预警模块中使用

```python
from ice_monitor.alert import fuse_alert_local

# 本地模型识别结果
local_labels = ['覆冰', '雪']

# 融合预警
alert = fuse_alert_local(
    ice_thickness=3.5,
    ice_ratio=0.15,
    local_labels=local_labels,
)
print(alert)
```

## 7. 实验结果

已有 `experiments/results/eval_report.json` 中的主要指标：

```text
覆冰厚度 MAE:  4.0635
覆冰厚度 RMSE: 8.2026
覆冰厚度 R2:   -0.3210
覆冰比值 MAE:  0.1274
覆冰比值 RMSE: 0.2241
Acc@0.1:       8.2%
Acc@0.2:       71.4%
Acc@0.3:       71.4%
Acc@0.5:       71.4%
Acc@1.0:       71.7%
总预测数:      19955
```

窗口指标：

```text
0-6h:   MAE=5.9200, Acc@0.2=60.15%
6-12h:  MAE=3.9678, Acc@0.2=69.90%
12-24h: MAE=3.1142, Acc@0.2=77.01%
24-48h: MAE=3.3720, Acc@0.2=78.58%
```

## 8. 预警模块

`ice_monitor/alert.py` 的 `fuse_alert()` 将时序模型预测和 VLM 标签融合：

- Level 0：正常。
- Level 1：注意。
- Level 2：预警。
- Level 3：紧急。

核心规则：

- 厚度大于 5mm 直接 Level 3。
- VLM 为 `yes` 且厚度大于 2mm，Level 3。
- VLM 为 `yes` 或厚度大于 2mm，Level 2。
- 厚度在 0.5 到 2mm，Level 1。
- VLM 不确定，Level 1。

### 8.1 本地模型预警

新增 `fuse_alert_local()` 函数，支持本地模型的多标签识别结果：

- 检测到"覆冰"且厚度 > 5mm，Level 3
- 检测到"覆冰"且厚度 > 2mm，Level 3
- 检测到任何覆冰相关标签（覆冰/雪/积雪/霜冻），Level 2
- 厚度 > 2mm，Level 2
- 厚度 0.5~2mm，Level 1
- 其他情况，Level 0

使用示例：

```python
from ice_monitor.alert import fuse_alert_local

alert = fuse_alert_local(
    ice_thickness=3.5,
    ice_ratio=0.15,
    local_labels=['覆冰', '雪'],
)
print(alert.level)       # 3
print(alert.level_name)  # 紧急
```

## 9. 常见问题

### 9.1 PowerShell 里 `Get-Content` 看源码乱码

这通常是控制台编码显示问题。Python 和浏览器按 UTF-8 读取时可以正常显示中文。

### 9.2 `No module named einops/kornia`

运行：

```powershell
python -m pip install -r requirements.txt
```

### 9.3 没有 `xformers`

当前代码已经有 fallback，不安装 `xformers` 也能运行。GPU 环境下如果想用原始高性能 attention，可以自行安装匹配当前 PyTorch/CUDA 版本的 `xformers`。

### 9.4 前端图表不显示

前端依赖 CDN 加载 Chart.js。确保网络能访问：

```text
https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js
```

### 9.5 直接双击 `demo/index.html` 数据不加载

用 HTTP 服务器打开：

```powershell
cd demo
python -m http.server 8000
```
