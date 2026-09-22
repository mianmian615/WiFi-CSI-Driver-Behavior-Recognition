# widgbed/datasets.py
# 主要功能
# 定义多域CSI数据集的加载方式。
# 支持多种CSI数据集（Widar3、CSIDA、ARIL等）。
# 提供可迭代数据集（CSIIterDataset），兼容PyTorch 2.4。
# 主要流程
# CSIIterDataset：实现__iter__，兼容PyTorch 2.4。
# MultipleEnvironmentCSI：多环境数据集封装，便于多域训练。
# CSI：最终对外暴露的数据集类。

import torch
from torch.utils.data import TensorDataset
import numpy as np

from widgbed.CSI.csi_domainset import dataset_csi_size
from widgbed.CSI.csi_dg import *

def get_dataset_class(dataset_name):
    """
    根据数据集名称返回对应的数据集类。
    """
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]

# 未用到的函数，保留以备后续扩展
# def num_environments(dataset_name):
#     return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class MultipleDomainDataset:
    """
    多域数据集基类，提供基本getitem和len实现。
    """
    def __getitem__(self, index):
        data, label = self.data[index], int(self.labels[index])
        return data, label
    def __len__(self):
        return len(self.data)

class CSIIterDataset(torch.utils.data.IterableDataset):
    """
    可迭代CSI数据集，兼容PyTorch 2.4。
    支持Widar3、CSIDA、ARIL等多种CSI数据集。
    """
    def __init__(self, args, domain, data_type) -> None:
        self.data_type = data_type
        self.data, self.labels = self.get_csidataset(args, domain)

    def get_csidataset(self, args, domain):
        """
        根据参数和domain加载CSI数据，返回数据和标签。
        优先加载缓存文件，若无则处理后保存。
        """
        import os
        if args.normalization:
            cache_dir = os.path.join(args.data_dir, 'cache_norm')
            cache_file = os.path.join(cache_dir, f"{domain}/Speed2_Inter1_norm.npz")
        else:
            cache_dir = os.path.join(args.data_dir, 'cache')
            cache_file = os.path.join(cache_dir, f"{domain}/Speed2_Inter1.npz")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        #os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_file):
            print('extracting:', cache_file)
            cache = np.load(cache_file)
            if args.data_type == 'amp':
                data = cache['amp']
            elif args.data_type == 'pha':
                data = cache['pha']
            elif args.data_type == 'amp+pha':
                data = np.concatenate((cache['amp'], cache['pha']), axis=2)
            else:
                raise ValueError('不支持的数据类型')
            labels = cache['labels']
            return data, labels
        
        
        dataset_dir = args.data_dir + "/" + "dat"
        if args.dataset == 'Widar3':
            amp, pha, labels, *_ = get_widar_csi(dataset_dir, domain)
        elif args.dataset == 'Peoplecounting':
            amp, pha, labels, *_ = get_Peoplecounting_csi(dataset_dir, domain)
        elif args.dataset == 'WiMANS':
            amp, pha, labels, *_ = get_wimans_csi(dataset_dir, domain)
        elif args.dataset == 'XRF55':
            amp, pha, labels, *_ = get_xrf55_csi(dataset_dir, domain)
        elif args.dataset == 'CSIDA':
            amp, pha, labels, *_ = get_CSIDA_csi(dataset_dir, domain)
        #elif args.dataset == 'WCBHAR':
            #amp, pha, labels, *_ = get_wcbhar_csi(dataset_dir, domain)
        elif args.dataset == 'NTUFIHumanID':
            amp, labels, *_ = get_ntufi_humanid_csi(dataset_dir, domain)
            pha = None  
        elif args.dataset == 'NTUFIHAR':
            amp, labels, *_ = get_ntufi_har_csi(dataset_dir, domain)
            pha = None
        elif args.dataset == 'ARIL':
            amp, pha, labels, *_ = get_ARIL_csi(dataset_dir, domain)
            amp = amp.reshape(amp.shape[0], 1, amp.shape[1], amp.shape[2])
            pha = pha.reshape(pha.shape[0], 1, pha.shape[1], pha.shape[2])
        else:
            raise ValueError('不支持的数据集类型')
        # 归一化amp和pha
        if args.normalization:
            amp = (amp - np.mean(amp, keepdims=True)) / (np.std(amp, keepdims=True) + 1e-8)
            if pha is not None:
                pha = (pha - np.mean(pha, keepdims=True)) / (np.std(pha, keepdims=True) + 1e-8)
        if args.data_type == 'amp':
            data = amp
        elif args.data_type == 'pha':
            data = pha
        elif args.data_type == 'amp+pha':
            data = np.concatenate((amp, pha), axis=2)
        else:
            raise ValueError('不支持的数据类型')
        # 保存缓存
        #os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, amp=amp, pha=pha, labels=np.asarray(labels))
        print('save cache:', cache_file)
        return data, labels

    def __getitem__(self, index: int):
        """
        支持索引访问，返回单个样本及标签。
        """
        data = self.data[index]
        label = self.labels[index]
        # 只对常见类型做int转换，其他类型直接返回
        if isinstance(label, np.ndarray) or isinstance(label, list):
            label = int(np.asarray(label).flatten()[0])
        elif isinstance(label, (int, float, np.integer, np.floating)):
            label = int(label)
        return data, label

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        """
        支持迭代访问，兼容PyTorch 2.4 IterableDataset。
        """
        for i in range(len(self.data)):
            label = self.labels[i]
            if isinstance(label, np.ndarray) or isinstance(label, list):
                label = int(np.asarray(label).flatten()[0])
            elif isinstance(label, (int, float, np.integer, np.floating)):
                label = int(label)
            yield self.data[i], label

class MultipleEnvironmentCSI(MultipleDomainDataset):
    """
    多环境CSI数据集封装，便于多域训练。
    """
    def __init__(self, args, data_type, dataset_transform):
        super().__init__()
        self.datasets = []
        if data_type is not None:
            if data_type == 'source':
                domains = args.source_domains
            if data_type == 'target':
                domains = args.target_domains
            for i in range(len(domains)):
                dataset = CSIIterDataset(args, domains[i], data_type)
                x = dataset.data
                y = dataset.labels
                d = np.array([i] * x.shape[0], dtype=int)
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y)
                d = torch.tensor(d)
                self.datasets.append(dataset_transform(x, y, d))
        self.input_shape = dataset_csi_size[args.dataset][args.data_type]
        self.num_classes = dataset_csi_size[args.dataset]['classes']

class CSI(MultipleEnvironmentCSI):
    """
    对外暴露的CSI多环境数据集类。
    """
    def __init__(self, args,  data_type=None):
        super(CSI, self).__init__(args, data_type, self.dataset_transform)
    def dataset_transform(self, x, y, domain):
        return TensorDataset(x, y, domain)