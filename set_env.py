"""
环境变量设置脚本
在Python代码中设置环境变量
"""

import os

if os.environ.get("MIMO_API_KEY"):
    print("已检测到环境变量：MIMO_API_KEY")
else:
    print("未检测到 MIMO_API_KEY，请先在当前终端设置环境变量。")
    print("示例：export MIMO_API_KEY='your_api_key'")
