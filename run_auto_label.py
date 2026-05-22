"""
数据标注工具 - 直接运行版本
无需命令行参数，直接在PyCharm中运行
"""

from tools.auto_label import MultiLabelAutoLabeler, convert_to_training_format
import os

# 配置参数
IMAGE_DIR = "data/imagine"
OUTPUT_FILE = "multi_label_results.json"
MODEL_NAME = "mimo-v2.5"
API_FORMAT = "openai"
BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"

def main():
    # 检查是否已有标注结果
    if os.path.exists(OUTPUT_FILE):
        print(f"标注结果文件已存在: {OUTPUT_FILE}")
        choice = input("是否重新标注? (y/n): ").strip().lower()
        if choice != 'y':
            print("跳过标注，转换为训练格式...")
            convert_to_training_format(OUTPUT_FILE, "data/labels")
            return

    # 运行标注
    print(f"开始标注图像目录: {IMAGE_DIR}")
    labeler = MultiLabelAutoLabeler(
        model=MODEL_NAME,
        api_format=API_FORMAT,
        base_url=BASE_URL,
    )
    labeler.label_folder(IMAGE_DIR, OUTPUT_FILE)

    # 转换为训练格式
    print("\n转换为训练格式...")
    convert_to_training_format(OUTPUT_FILE, "data/labels")

if __name__ == "__main__":
    main()
