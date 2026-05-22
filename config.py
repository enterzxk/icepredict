"""
API配置文件
支持多种VLM API提供商：小米MiMo、阿里云DashScope等
"""

import os
from typing import Optional

# =========== API提供商配置 ===========

# 小米MiMo API配置
MIMO_CONFIG = {
    "base_url": "https://token-plan-cn.xiaomimimo.com/v1",  # 小米MiMo API地址
    "model": "mimo-v2.5",  # 图像标注使用 MiMo 2.5
    "api_key_env": "MIMO_API_KEY",  # 环境变量名
}

# 阿里云DashScope API配置
DASHSCOPE_CONFIG = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-vl-plus",
    "api_key_env": "DASHSCOPE_API_KEY",
}

# OpenAI API配置（如果有）
OPENAI_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4-vision-preview",
    "api_key_env": "OPENAI_API_KEY",
}

# =========== 默认配置 ===========
# 设置为 "mimo"、"dashscope" 或 "openai"
DEFAULT_PROVIDER = "mimo"

def get_api_config(provider: str = None) -> dict:
    """
    获取API配置

    Args:
        provider: API提供商名称，可选 "mimo"、"dashscope"、"openai"
                  如果为None，使用默认提供商

    Returns:
        dict: 包含 base_url、model、api_key 的配置
    """
    if provider is None:
        provider = DEFAULT_PROVIDER

    configs = {
        "mimo": MIMO_CONFIG,
        "dashscope": DASHSCOPE_CONFIG,
        "openai": OPENAI_CONFIG,
    }

    if provider not in configs:
        raise ValueError(f"不支持的API提供商: {provider}，可选: {list(configs.keys())}")

    config = configs[provider]

    # 从环境变量获取API Key
    api_key = os.environ.get(config["api_key_env"], "")

    return {
        "base_url": config["base_url"],
        "model": config["model"],
        "api_key": api_key,
        "api_key_env": config["api_key_env"],
    }


def get_api_key(provider: str = None) -> str:
    """
    获取API Key

    Args:
        provider: API提供商名称

    Returns:
        str: API Key
    """
    config = get_api_config(provider)
    api_key = config["api_key"]

    if not api_key:
        raise ValueError(
            f"缺少API Key。请设置环境变量 {config['api_key_env']}\n"
            f"例如: export {config['api_key_env']}='your_api_key'"
        )

    return api_key
