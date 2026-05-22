"""
数据集划分工具 - 直接运行版本
无需命令行参数，直接在PyCharm中运行
"""

from tools.split_dataset import load_labels, stratified_split, save_split_info, create_symlinks
import os

# 配置参数
LABEL_FILE = "data/labels/binary_training_labels.json"
OUTPUT_DIR = "data/binary_dataset"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
CREATE_SYMLINKS = False  # 二分类训练直接读取JSON中的原图路径，默认不复制/链接几千张图片

def main():
    create_symlinks_enabled = CREATE_SYMLINKS

    # 检查标注文件
    if not os.path.exists(LABEL_FILE):
        print(f"错误: 标注文件不存在: {LABEL_FILE}")
        print("请先运行 run_auto_label.py 生成标注")
        return

    # 加载数据
    print(f"加载标注数据: {LABEL_FILE}")
    data = load_labels(LABEL_FILE)
    print(f"总样本数: {len(data)}")

    # 划分数据集
    print("\n划分数据集...")
    train_data, val_data, test_data = stratified_split(
        data,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        SEED,
    )

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建符号链接
    if create_symlinks_enabled:
        print("\n创建数据集目录...")
        try:
            create_symlinks(train_data, OUTPUT_DIR, "train")
            create_symlinks(val_data, OUTPUT_DIR, "val")
            create_symlinks(test_data, OUTPUT_DIR, "test")
            print("数据集目录创建完成")
        except OSError as e:
            print(f"创建符号链接失败: {e}")
            print("请以管理员身份运行PyCharm，或设置 CREATE_SYMLINKS = False")
            create_symlinks_enabled = False

    # 保存划分信息
    save_split_info(train_data, val_data, test_data, OUTPUT_DIR)

    print(f"\n数据集划分完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练集: {len(train_data)} 张")
    print(f"验证集: {len(val_data)} 张")
    print(f"测试集: {len(test_data)} 张")

if __name__ == "__main__":
    main()
