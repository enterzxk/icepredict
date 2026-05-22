"""PyCharm-friendly entrypoint for MiMo patch auto-labeling."""

import os
from argparse import Namespace

from config import get_api_config
from tools.auto_label_patches import auto_label_patches


def main():
    config = get_api_config("mimo")
    args = Namespace(
        patch_csv="data/patch_candidates/patch_candidates.csv",
        output_csv="data/patch_candidates/patch_candidates_mimo.csv",
        api_format="anthropic",
        model=config["model"],
        base_url=config["base_url"],
        api_key_env=config["api_key_env"],
        api_key=os.environ.get(config["api_key_env"], ""),
        timeout=60,
        max_items=0,
        retries=3,
        retry_sleep=2.0,
        sleep=0.3,
        save_every=10,
        overwrite=False,
        dry_run=False,
    )

    print("=" * 60)
    print("Patch视觉模型自动标注 - MiMo")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"API格式: {args.api_format}")
    print(f"API地址: {args.base_url}")
    print(f"API Key环境变量: {args.api_key_env}")
    print(f"输入CSV: {args.patch_csv}")
    print(f"输出CSV: {args.output_csv}")
    print("策略: 保守标注；看不清/模糊/夜间/雨雾/反光 => uncertain")
    print("=" * 60)

    if not args.api_key:
        raise ValueError(
            f"缺少API Key。请先设置环境变量 {args.api_key_env}，"
            f"例如: export {args.api_key_env}='your_api_key'"
        )

    summary = auto_label_patches(args)
    print("\n完成。下一步转换训练集：")
    print(
        "/home/yndx/miniconda3/envs/a40_env/bin/python "
        "tools/convert_patch_labels.py "
        "--patch-csv data/patch_candidates/patch_candidates_mimo.csv "
        "--output-dir data/patch_dataset_mimo --require-exists"
    )
    return summary


if __name__ == "__main__":
    main()
