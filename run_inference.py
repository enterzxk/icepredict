"""
推理测试脚本 - 直接运行版本
无需命令行参数，直接在PyCharm中运行
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ice_monitor.local_detector import LocalIceDetector
from ice_monitor.alert import fuse_alert_local

def main():
    # 配置参数
    CHECKPOINT = "weights/ice_classifier/best_stage2.pth"
    TEST_IMAGE = "data/imagine/CrRRgmdL00SAVdkbAAI78pMVqMk536.jpg"  # 测试图像路径
    THRESHOLD = 0.5

    # 检查checkpoint
    if not os.path.exists(CHECKPOINT):
        print(f"错误: 模型checkpoint不存在: {CHECKPOINT}")
        print("请先运行 run_train.py 训练模型")
        return

    # 检查测试图像
    if not os.path.exists(TEST_IMAGE):
        print(f"错误: 测试图像不存在: {TEST_IMAGE}")
        print("请修改 TEST_IMAGE 变量指向一个存在的图像文件")
        return

    # 创建检测器
    print("="*60)
    print("覆冰图像识别推理测试")
    print("="*60)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"测试图像: {TEST_IMAGE}")
    print(f"阈值: {THRESHOLD}")
    print("="*60)

    detector = LocalIceDetector(
        checkpoint_path=CHECKPOINT,
        threshold=THRESHOLD,
    )

    # 单张图像推理
    print("\n推理结果:")
    result = detector.detect(TEST_IMAGE)
    print(f"  图像: {result['image_path']}")
    print(f"  标签: {result['label']}")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  详细概率:")
    for label, prob in result['details'].items():
        print(f"    {label}: {prob:.3f}")

    # 测试预警融合
    print("\n预警融合测试:")
    alert = fuse_alert_local(
        ice_thickness=3.5,
        ice_ratio=0.15,
        local_labels=result['labels'],
    )
    print(f"  预警等级: {alert.level}")
    print(f"  预警名称: {alert.level_name}")
    print(f"  预警原因: {alert.reason}")

    # 测试批量推理
    print("\n批量推理测试（前5张图像）:")
    image_dir = "data/imagine"
    if os.path.exists(image_dir):
        from pathlib import Path
        image_extensions = {".jpg", ".jpeg", ".png"}
        images = [
            str(f.resolve())
            for f in Path(image_dir).glob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ][:5]

        for img_path in images:
            result = detector.detect(img_path)
            print(f"  {os.path.basename(img_path)}: {result['label']} ({result['confidence']:.3f})")

    print("\n测试完成！")

if __name__ == "__main__":
    main()
