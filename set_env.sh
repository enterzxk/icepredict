#!/bin/bash
# 设置环境变量脚本
# 使用方法: source set_env.sh

if [ -z "${MIMO_API_KEY:-}" ]; then
  echo "未检测到 MIMO_API_KEY，请先执行："
  echo "  export MIMO_API_KEY='your_api_key'"
  return 1 2>/dev/null || exit 1
fi

echo "已检测到环境变量：MIMO_API_KEY"
