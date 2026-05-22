import os
import random
import shutil
from typing import List

def process_csv_files(folder_path: str) -> tuple[List[str], List[str], List[str]]:
    """
    处理指定文件夹下的CSV文件，提取非零文件、前置零文件、后置零文件列表
    若前置/后置零文件数量不足，从所有_0_00.csv文件中随机补充至等量
    
    Args:
        folder_path: CSV文件所在文件夹路径
    
    Returns:
        nonzero_csv: 非_0_00.csv结尾的文件列表（按文件名排序）
        pre_zero_csv: 第一个非零文件前的等量_0_00.csv文件列表（不足则随机补充）
        post_zero_csv: 最后一个非零文件后的等量_0_00.csv文件列表（不足则随机补充）
    """
    # 1. 获取文件夹下所有CSV文件，按文件名排序
    all_csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and "_覆冰厚度_" in file:  # 过滤目标CSV文件
            all_csv_files.append(file)
    all_csv_files.sort()  # 按文件名排序（保证时间/编号顺序）
    
    # 2. 筛选非零文件列表（结尾不是_0_00.csv）
    nonzero_csv = [f for f in all_csv_files if not f.endswith("_0_00.csv")]
    nonzero_count = len(nonzero_csv)
    if nonzero_count == 0:
        return [], [], []  # 无数据时返回空列表
    
    # 3. 提取所有_0_00.csv文件（用于后续补充）
    all_zero_csv = [f for f in all_csv_files if f.endswith("_0_00.csv")]
    
    # 4. 提取前置零文件（第一个非零文件前的_0_00.csv）
    first_nonzero_idx = all_csv_files.index(nonzero_csv[0])
    pre_zero_csv = []
    # 从第一个非零文件往前找，收集_0_00.csv
    for idx in range(first_nonzero_idx - 1, -1, -1):
        if len(pre_zero_csv) >= nonzero_count:
            break
        if all_csv_files[idx].endswith("_0_00.csv"):
            pre_zero_csv.append(all_csv_files[idx])
    pre_zero_csv = pre_zero_csv[::-1]  # 反转恢复正序
    
    # 5. 补充前置零文件（数量不足时随机选取）
    if len(pre_zero_csv) < nonzero_count:
        # 排除已提取的前置零文件，避免重复
        candidate_zero = [f for f in all_zero_csv if f not in pre_zero_csv]
        # 随机选取需要补充的数量
        need补充 = nonzero_count - len(pre_zero_csv)
        # 若候选不足，允许重复（或根据需求调整，此处默认允许）
        if len(candidate_zero) < need补充:
            补充_files = random.choices(candidate_zero, k=need补充)
        else:
            补充_files = random.sample(candidate_zero, need补充)
        pre_zero_csv += 补充_files
    
    # 6. 提取后置零文件（最后一个非零文件后的_0_00.csv）
    last_nonzero_idx = all_csv_files.index(nonzero_csv[-1])
    post_zero_csv = []
    # 从最后一个非零文件往后找，收集_0_00.csv
    for idx in range(last_nonzero_idx + 1, len(all_csv_files)):
        if len(post_zero_csv) >= nonzero_count:
            break
        if all_csv_files[idx].endswith("_0_00.csv"):
            post_zero_csv.append(all_csv_files[idx])
    
    # 7. 补充后置零文件（数量不足时随机选取）
    if len(post_zero_csv) < nonzero_count:
        # 排除已提取的后置零文件，避免重复
        candidate_zero = [f for f in all_zero_csv if f not in post_zero_csv]
        # 随机选取需要补充的数量
        need补充 = nonzero_count - len(post_zero_csv)
        if len(candidate_zero) < need补充:
            补充_files = random.choices(candidate_zero, k=need补充)
        else:
            补充_files = random.sample(candidate_zero, need补充)
        post_zero_csv += 补充_files
    
    return nonzero_csv, pre_zero_csv, post_zero_csv

def split_train_test(file_list: List[str], test_ratio: float = 0.1) -> tuple[List[str], List[str]]:
    """
    按指定比例拆分训练/测试数据（随机选取，保证数据不重复）
    
    Args:
        file_list: 待拆分的文件列表
        test_ratio: 测试数据比例（默认10%）
    
    Returns:
        train_files: 训练文件列表
        test_files: 测试文件列表
    """
    # 固定随机种子（可选，保证每次拆分结果一致）
    random.seed(123)
    
    # 计算测试数据数量（向上取整，避免0个测试数据）
    test_count = max(1, int(len(file_list) * test_ratio))
    
    # 随机选取测试数据
    test_files = random.sample(file_list, test_count)
    
    # 训练数据 = 总数据 - 测试数据
    train_files = [f for f in file_list if f not in test_files]
    
    return train_files, test_files

def copy_files_to_target(source_dir: str, target_dir: str, file_list: List[str]):
    """
    将指定文件列表从源目录拷贝到目标目录（自动创建目标目录）
    
    Args:
        source_dir: 源文件所在目录
        target_dir: 目标目录路径
        file_list: 需要拷贝的文件名列表
    """
    # 确保目标目录存在，不存在则递归创建
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历文件列表执行拷贝
    for filename in file_list:
        # 拼接源文件和目标文件的完整路径
        src_file = os.path.join(source_dir, filename)
        dst_file = os.path.join(target_dir, filename)
        
        # 检查源文件是否存在
        if not os.path.exists(src_file):
            print(f"警告：源文件不存在，跳过拷贝 -> {src_file}")
            continue
        
        # 执行拷贝（copy2保留文件元数据，如需覆盖可添加参数：copy2(src, dst)）
        try:
            shutil.copy2(src_file, dst_file)
            print(f"成功拷贝：{filename} -> {target_dir}")
        except Exception as e:
            print(f"错误：拷贝文件 {filename} 失败，原因 -> {str(e)}")

# ------------------- 主执行逻辑 -------------------
if __name__ == "__main__":
    # 原始数据目录（可根据需要修改）
    source_folder = r"D:\work\DianLi\IcethickRegression\data\data\分片数据_前6_当前_后72\CC8839"
    
    # 自动构建训练/测试目标目录（按规则：_平衡_训练 / _平衡_测试）
    # 拆分路径：获取父目录和子目录名称
    parent_dir = os.path.dirname(source_folder)  # 父目录：分片数据_前6_当前_后72
    sub_dir = os.path.basename(source_folder)    # 子目录：CC8839
    
    # 构建训练/测试目标目录
    train_parent_dir = parent_dir + "_平衡_训练"
    test_parent_dir = parent_dir + "_平衡_测试"
    train_target_folder = os.path.join(train_parent_dir, sub_dir)
    test_target_folder = os.path.join(test_parent_dir, sub_dir)
    
    # 1. 处理CSV文件，获取三个核心列表
    nonzero_csv, pre_zero_csv, post_zero_csv = process_csv_files(source_folder)
    
    # 2. 按10%比例拆分每个列表的训练/测试数据
    nonzero_train, nonzero_test = split_train_test(nonzero_csv, test_ratio=0.1)
    pre_zero_train, pre_zero_test = split_train_test(pre_zero_csv, test_ratio=0.1)
    post_zero_train, post_zero_test = split_train_test(post_zero_csv, test_ratio=0.1)
    
    # 3. 合并训练/测试文件列表
    all_train_files = pre_zero_train + nonzero_train + post_zero_train
    all_test_files = pre_zero_test + nonzero_test + post_zero_test
    
    # 4. 执行训练数据拷贝
    print("===== 开始拷贝训练数据 =====")
    copy_files_to_target(source_folder, train_target_folder, all_train_files)
    
    # 5. 执行测试数据拷贝
    print("\n===== 开始拷贝测试数据 =====")
    copy_files_to_target(source_folder, test_target_folder, all_test_files)
    
    # 6. 输出统计信息
    print("\n=== 拷贝完成 ===")
    print(f"源目录：{source_folder}")
    print(f"训练数据目标目录：{train_target_folder}")
    print(f"测试数据目标目录：{test_target_folder}")
    print("-" * 50)
    print(f"【非零文件】总数量：{len(nonzero_csv)} | 训练：{len(nonzero_train)} | 测试：{len(nonzero_test)}")
    print(f"【前置零文件】总数量：{len(pre_zero_csv)} | 训练：{len(pre_zero_train)} | 测试：{len(pre_zero_test)}")
    print(f"【后置零文件】总数量：{len(post_zero_csv)} | 训练：{len(post_zero_train)} | 测试：{len(post_zero_test)}")
    print("-" * 50)
    print(f"训练数据总计：{len(all_train_files)} 个文件")
    print(f"测试数据总计：{len(all_test_files)} 个文件")