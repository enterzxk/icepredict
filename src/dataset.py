import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import json
import pickle
import time
from datetime import datetime

def time_to_timestamp(time_str):
    """将"2024-01-01 00:05:30"转为秒级时间戳（整数）"""
    # 解析时间字符串为datetime对象
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    # 转换为时间戳（秒级）
    timestamp = int(time.mktime(dt.timetuple()))
    return timestamp

# class FubingDataset(Dataset):
#     def __init__(self):
#         if not os.path.exists('data/data/merged_data.csv'):
#             files = glob('data/data/*/*.csv')
#             dfs = []
#             keys = ['供电局', '电压等级', '线路名', '相别', '覆冰厚度', '覆冰比值', '拉力', '最小拉力', '最大拉力风偏角', \
#                         '最小拉力风偏角', '温度', '湿度', '采集时间', '杆塔单元', '终端编号', '时间']
#             for file_path in files:
#                 df = pd.read_csv(file_path, encoding='utf-8')
#                 # 相别包含[A相, B相, 地线], 此处选择地线
#                 df = df[df[keys[3]].str.contains('地')][[keys[4], keys[5],keys[10],keys[11], keys[12],keys[14],keys[15]]]
#                 dfs.append(df)
            
#             csv_data = pd.concat(dfs, ignore_index=True)
#             csv_data.to_csv('data/data/merged_data.csv', index=False, encoding='utf-8')
#         else:
#             csv_data = pd.read_csv('data/data/merged_data.csv', encoding='utf-8')

#         types = np.unique(csv_data['终端编号'].values)
#         types_to_number = {type: i for i, type in enumerate(types)}
#         # csv_data['终端编号'] = csv_data['终端编号'].map(types_to_number)

#         max_time = 3600 * 2
#         min_time = -3600 * 24 * 3
#         chunk_size = 1000
#         self.type_data = {}
#         for type in types:
#             type_data = {}
#             type_data['data'] = csv_data[csv_data['终端编号'] == type]
#             times = pd.to_datetime(type_data[keys[-1]], format='%Y-%m-%d %H:%M:%S').values
#             chunks = np.ceil(len(times) / chunk_size).astype(int)
#             valid_indices = []
#             pre_indices = []
#             post_indices = []
#             for i in range(chunks):
#                 si = i * chunk_size
#                 chunk_times = times[i*chunk_size:(i+1)*chunk_size]
#                 times_diff = np.float32((chunk_times[:,None] - times[None,:]) / 10**9) # 转换为秒
#                 mask = times_diff >= max_time
#                 valid_mask = np.sum(mask,axis=1) > 0
#                 for j in range(len(valid_mask)):
#                     if valid_mask[j]:
#                         valid_indices.append(si + j)
#                         pre_indices.append(np.where(times_diff[j,:] < max_time and times_diff[j,:]>0))
#                         post_indices.append(np.where(times_diff[j,:] >= min_time and times_diff[j,:]<0))
#             type_data['valid_indices'] = valid_indices
#             type_data['pre_indices'] = pre_indices
#             type_data['post_indices'] = post_indices
#             self.type_data[type] = type_data
            
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data.iloc[idx]

class FubingDataset(object):
    def __init__(self,batch_size=32):
        self.batch_size = batch_size
        keys = ['供电局', '电压等级', '线路名', '相别', '覆冰厚度', '覆冰比值', '拉力', '最小拉力', '最大拉力风偏角', \
                            '最小拉力风偏角', '温度', '湿度', '采集时间', '杆塔单元', '终端编号', '时间']
        if not os.path.exists('data/data/data_dict.pkl'):
            if not os.path.exists('data/data/merged_data.csv'):
                files = glob('data/data/*/*.csv')
                dfs = []
                for file_path in files:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    # 相别包含[A相, B相, 地线], 此处选择地线
                    df = df[df[keys[3]].str.contains('地')][[keys[4], keys[5],keys[10],keys[11],keys[15],keys[14]]]
                    dfs.append(df)
                
                csv_data = pd.concat(dfs, ignore_index=True)
                csv_data.to_csv('data/data/merged_data.csv', index=False, encoding='utf-8')
            else:
                csv_data = pd.read_csv('data/data/merged_data.csv', encoding='utf-8')

            types = np.unique(csv_data['终端编号'].values)
            

            max_time = 3600 * 6
            min_time = -3600 * 24 * 2
            chunk_size = 1000
            self.type_data = {}
            for type in types:
                type_data = {}
                _type_df = csv_data[csv_data['终端编号'] == type]
                _type_df[keys[-1]] = pd.to_datetime(_type_df[keys[-1]], format='%Y-%m-%d %H:%M:%S')
                type_data['data'] = _type_df.values
                times = _type_df[keys[-1]].values
                chunks = np.ceil(len(times) / chunk_size).astype(int)
                valid_indices = []
                pre_indices = []
                post_indices = []
                for i in range(chunks):
                    si = i * chunk_size
                    chunk_times = times[i*chunk_size:(i+1)*chunk_size]
                    times_diff = np.float32((chunk_times[:,None] - times[None,:]) / 10**9) # 转换为秒
                    mask = (times_diff >= max_time) | (times_diff < 0) # 最少追溯到2小时之前，且最少有一条记录在未来
                    valid_mask = np.sum(mask,axis=1) > 1
                    for j in range(len(valid_mask)):
                        if valid_mask[j]:
                            valid_indices.append(si + j)
                            pre_indices.append(np.where((times_diff[j,:] < max_time) & (times_diff[j,:]>0)))
                            post_indices.append(np.where((times_diff[j,:] >= min_time) & (times_diff[j,:]<0)))
                type_data['valid_indices'] = valid_indices
                type_data['pre_indices'] = pre_indices
                type_data['post_indices'] = post_indices
                self.type_data[type] = type_data
            
            with open("data/data/data_dict.pkl", "wb") as f:
                pickle.dump(self.type_data, f)

        else:
            with open("data/data/data_dict.pkl", "rb") as f:
                self.type_data = pickle.load(f)
        
        self.types_to_number = {type: i for i, type in enumerate(self.type_data.keys())}
        self.type_lengths = [len(self.type_data[type]['valid_indices']) for type in self.type_data]
        self.total_length = sum(self.type_lengths)
        self.num_types = len(self.type_data)
        self.types = list(self.type_data.keys())
        self.index = 0
        self.indices = list(range(self.total_length))
        self.shuffle()
    
    def shuffle(self):
        np.random.shuffle(self.indices)
    
    def __iter__(self):
        return self

    def __next__(self):
        # Implement batch fetching logic here
        if self.index >= self.total_length:
            self.index = 0
            raise StopIteration
        batch_size = min(self.total_length - self.index,self.batch_size)
        using_index = self.indices[self.index:self.index + batch_size]
        self.index += batch_size
        batch_data = []
        for index in using_index:
            # Determine which type and index within that type
            cumulative_length = 0
            for i,type_length in enumerate(self.type_lengths):
                if index < cumulative_length + type_length:
                    type = self.types[i]
                    local_index = index - cumulative_length
                    valid_idx = self.type_data[type]['valid_indices'][local_index]
                    pre_indices = self.type_data[type]['pre_indices'][local_index]
                    post_indices = self.type_data[type]['post_indices'][local_index]
                    _type_data = self.type_data[type]['data']
                    batch_data.append([valid_idx, _type_data[valid_idx], _type_data[pre_indices], _type_data[post_indices]])
                    # batch_data.append([type, valid_idx, pre_indices,post_indices])
                    break
                cumulative_length += type_length
        
        # return batch_data
        batch_pre, batch_post = [],[]
        for item in batch_data:
            valid_idx, valid_data, pre_data, post_data = item
            pre_data = np.concatenate([pre_data,valid_data[None,:]], axis=0)
            pre_data, pre_times = np.float32(pre_data[:, :-2]), pre_data[:, -2]
            pre_times = np.array([time.mktime(t.timetuple()) for t in pre_times],dtype=np.float32)
            pre_data = np.concatenate([pre_data, pre_times[:,None]], axis=-1) # (n,5)
            
            post_data, post_times = np.float32(post_data[:, :-2]), post_data[:, -2]
            post_times = np.array([time.mktime(t.timetuple()) for t in post_times],dtype=np.float32)
            post_data = np.concatenate([post_data, post_times[:,None]], axis=-1) # (m,5)

            init_time = pre_data[0,-1]
            pre_data[:,-1] = (pre_data[:,-1] - init_time) / 3600.0 / 24.0 # 转换为天
            post_data[:,-1] = (post_data[:,-1] - init_time) / 3600.0 / 24.0

            pre_data[:,-2] = pre_data[:,-2] / 100.0 # 湿度归一化
            post_data[:,-2] = post_data[:,-2] / 100.0

            batch_pre.append(pre_data)
            batch_post.append(post_data)
        return batch_pre, batch_post
    

if __name__ == "__main__":
    dataset = FubingDataset(batch_size=2)

    # for batch in dataset:
    #     print(batch)
    #     break

    for i,(pre_data, post_data) in enumerate(dataset):
        print(pre_data, post_data)
        print(pre_data[0].shape, post_data[0].shape)