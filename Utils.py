from EdfData import Channel
from SampleSequence import SampleSeq
from typing import List
import matplotlib.pyplot as plt
import os
import numpy as np


class Utils:

    @staticmethod
    def plot_multi_channel(samples: List[SampleSeq], time_length: int) -> None:
        """
        Parameters:
        ---------------
        samples: n个通道的数据,
        time_length: 要画图的时间长度，单位毫秒
        """
        plt.figure(figsize=(12, 10))
        total = len(samples)
        samples = Utils.slice_multi_channel_samples(samples, time_length)

        total_seconds = time_length // 1000
        x_ticks_point = [i * 500 for i in range(total_seconds + 1)]
        x_ticks_label = [str(i) for i in range(total_seconds + 1)]
        for i, sample in enumerate(samples):
            plt.subplot(total // 2, 2, i + 1)
            plt.ylim(-5*1e-5, 5*1e-5)
            plt.xlim(0, time_length // 2)
            plt.xticks(x_ticks_point, x_ticks_label)
            plt.plot(sample.sample, scalex=False, scaley=False)
            plt.title(f"{sample.channel_name}-{i + 1}")
        plt.show()
    
    @staticmethod
    def export_multi_channle_samples(save_dir: str, samples_list: List[List[SampleSeq]], time_length: int) -> None:
        """
        导出图片
        Parameters:
        -----------------
        save_dir: 保存的目录
        samples_list: n_samples * m_channle 的数据，n个采样数据，每个采样数据有m个通道
        time_length: 要画图的时间长度
        """
        if not Utils.check_dir(save_dir):
            raise Exception(f"目录{save_dir}不存在")
        for samples in samples_list: # 取出每个采样数据，采样数据包含多个通道的
            file_path = os.path.join(save_dir, f"{samples[0].mid_idx}.jpg")
            plt.figure(figsize=(12, 10))
            samples = Utils.slice_multi_channel_samples(samples, time_length)
            total = len(samples)
            total_seconds = samples[0].time_length // 1000
            x_ticks_point = [i * 500 for i in range(total_seconds + 1)]
            x_ticks_label = [str(i) for i in range(total_seconds + 1)]
            for i, sample in enumerate(samples):
                plt.subplot(total // 2, 2, i + 1)
                y_min = sample.min * 1.5 * 1000000
                y_max = sample.max * 1.5 * 1000000
                y_min_label = int(y_min) // 10 * 10
                y_max_label = int(y_max) // 10 * 10
                plt.ylim(y_min, y_max)
                plt.yticks([y_min_label, -80, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', '80', str(y_max_label)])
                plt.gca().invert_yaxis()
                plt.xlim(0, sample.length)
                data = sample.sample * 1000000
                plt.xticks(x_ticks_point, x_ticks_label)
                plt.plot(data, scalex=False, scaley=False)
                plt.title(f"{sample.channel_name}-{i + 1}")
            plt.savefig(file_path)

        

    @staticmethod
    def check_dir(path: str) -> bool:
        """"
        检查目录是否存在，不存在则创建
        Parameters:
        -----------------    
        path: 要检查的目录

        Return
        ----------------
        true: 存在
        false: 不存在
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            return True
        except:
            return False


    @staticmethod
    def slice_multi_channel_samples(samples: List[SampleSeq], time_length: int) -> List[SampleSeq]:
        """
        根据给定的数据的中心点，重新采样到指定的时间长度
        Parameters:
        ---------------
        samples: n个通道的采样数据
        time_length: 时间长度

        Return
        --------------
        返回重新采样后的数据
        """
        return [Utils.slice_one_channel_sample(sample, time_length) for sample in samples]
    
    @staticmethod
    def transpose_matrix(matrix: List[list]) -> List[list]:
    # 获取矩阵的行数和列数
        rows = len(matrix)
        cols = len(matrix[0])

        # 创建一个新的二维列表来存储对调后的矩阵
        transposed_matrix = [[None] * rows for _ in range(cols)]

        # 遍历原始矩阵的行和列，并将其对调存储到新的矩阵中
        for i in range(rows):
            for j in range(cols):
                transposed_matrix[j][i] = matrix[i][j]

        return transposed_matrix

    @staticmethod
    def slice_one_channel_sample(sample: SampleSeq, time_length: int) -> SampleSeq:
        """
        根据给定的数据的中心点，重新采样到指定的时间长度
        Parameters:
        ---------------
        sample: 1个通道的采样数据
        time_length: 时间长度, 单位毫秒

        Return
        ---------------
        返回重新采样后的数据
        """
        step = time_length // 4
        mid = sample.mid_idx
        channel = sample.channel  # type: Channel
        new_data = channel.data[mid-step: mid + step] # 这里有可能左右边界越界，但是对于标注数据来说没关系，对于训练数据决不允许越界，进来的数据已经是非越界数据，此处不用做特殊处理
        return SampleSeq(sample.event_idx, time_length, new_data, channel, sample.bias, sample.type)
    
    @staticmethod
    def export_sample(sampleseq_list: List[List[SampleSeq]]) -> List[np.ndarray]:
        """
        返回整合完的n通道数据

        Args:
            sampleseq_list (List[List[SampleSeq]]): 
                channels_sampler: [sample nums, channel nums]

        Returns:
        ---------------
        返回重新采样后的数据
        """
        data_all = []
        for i in sampleseq_list:
            data_all.append(np.array([j.sample for j in i]))
        return data_all