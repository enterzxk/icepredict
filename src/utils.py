import time
from datetime import datetime

def time_to_timestamp(time_str):
    """将"2024-01-01 00:05:30"转为秒级时间戳（整数）"""
    # 解析时间字符串为datetime对象
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    print(dt,dt.timetuple(),time.mktime(dt.timetuple()))
    # 转换为时间戳（秒级）
    timestamp = int(time.mktime(dt.timetuple()))
    return timestamp


if __name__ == "__main__":
    t1_str = "2024-01-23 22:14:15"
    t2_str = "2024-01-24 00:14:45"
    t1 = time_to_timestamp(t1_str)
    t2 = time_to_timestamp(t2_str)
    print(t1,t2)
    print(t2-t1)