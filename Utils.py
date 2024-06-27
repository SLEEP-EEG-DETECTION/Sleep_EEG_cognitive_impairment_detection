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
        if samples[0].time_length != time_length:
            samples = Utils.slice_multi_channel_samples(samples, time_length)

        total_seconds = time_length // 1000
        x_ticks_point = [i * 500 for i in np.arange(0, total_seconds + 0.5, 0.5)]
        x_ticks_label = [str(i) for i in np.arange(0, total_seconds + 0.5, 0.5)]
        for i, sample in enumerate(samples):
            plt.subplot(total // 2, 2, i + 1)
            y_min = sample.min * 1.5 * 1000000
            y_max = sample.max * 1.5 * 1000000
            y_min_label = int(y_min) // 10 * 10
            y_max_label = int(y_max) // 10 * 10
            plt.ylim(y_min, y_max)
            if y_min_label >= -80 and y_max_label <= 80:
                plt.yticks([y_min_label, -50, 0, 50, y_max_label], [str(y_min_label), '-50', '0', '50', str(y_max_label)])
            elif y_min_label >= -80 and y_max_label > 80:
                plt.yticks([y_min_label, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-50', '0', '50', '80', str(y_max_label)])
            elif y_min_label < -80 and y_max_label <= 80:
                plt.yticks([y_min_label, -80, -50, 0, 50, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', str(y_max_label)])
            else:
                plt.yticks([y_min_label, -80, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', '80', str(y_max_label)])
            plt.gca().invert_yaxis()
            plt.xlim(0, sample.length)
            data = sample.sample * 1000000
            plt.xticks(x_ticks_point, x_ticks_label)
            plt.plot(data, scalex=False, scaley=False)
            plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')
            # 添加浅色的零线
            plt.axhline(0, color='lightgrey', linewidth=0.5)  # 添加y=0水平线
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (μV)')
            plt.title(f"{sample.channel_name}-{i + 1}")
        plt.show()
    

    @staticmethod
    def plot_multi_channel_numpy(sample: np.ndarray, time_length: int, channle_names: List[str]) -> None:
        """
        画图，画单个样本的4个通道的数据的图
        Parameters:
        ---------------
        sample: 一个样本的数据,numpy数组
        time_length: 要画图的时间长度
        """
        plt.figure(figsize=(12, 10))
        total = len(sample)

        total_seconds = time_length / 1000
        x_ticks_point = [i * 500 for i in np.arange(0, total_seconds + 0.5, 0.5)]
        x_ticks_label = [str(i) for i in np.arange(0, total_seconds + 0.5, 0.5)]
        for i, s in enumerate(sample):
            plt.subplot(total // 2, 2, i + 1)
            y_min = np.min(s) * 1.5 * 1000000
            y_max = np.max(s) * 1.5 * 1000000
            y_min_label = int(y_min) // 10 * 10
            y_max_label = int(y_max) // 10 * 10
            plt.ylim(y_min, y_max)
            if y_min_label >= -80 and y_max_label <= 80:
                plt.yticks([y_min_label, -50, 0, 50, y_max_label], [str(y_min_label), '-50', '0', '50', str(y_max_label)])
            elif y_min_label >= -80 and y_max_label > 80:
                plt.yticks([y_min_label, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-50', '0', '50', '80', str(y_max_label)])
            elif y_min_label < -80 and y_max_label <= 80:
                plt.yticks([y_min_label, -80, -50, 0, 50, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', str(y_max_label)])
            else:
                plt.yticks([y_min_label, -80, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', '80', str(y_max_label)])
            plt.gca().invert_yaxis()
            plt.xlim(0, time_length // 2)
            data = s * 1000000
            plt.xticks(x_ticks_point, x_ticks_label)
            plt.plot(data, scalex=False, scaley=False)
            plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')
            # 添加浅色的零线
            plt.axhline(0, color='lightgrey', linewidth=0.5)  # 添加y=0水平线
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (μV)')
            plt.title(f"{channle_names[i]}")
        plt.show()

    @staticmethod
    def export_multi_numpy(save_dir: str, samples_list: np.ndarray, time_length: int, channle_names: List[str]) -> None:
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
        for i, samples in enumerate(samples_list): # 取出每个采样数据，采样数据包含多个通道的
            file_path = os.path.join(save_dir, f"{i}.jpg")
            plt.figure(figsize=(12, 10))
            total = len(samples)
            total_seconds = len(samples[0]) * 2 // 1000
            x_ticks_point = [i * 500 for i in range(total_seconds + 1)]
            x_ticks_label = [str(i) for i in range(total_seconds + 1)]
            for i, sample in enumerate(samples):  # samples: 4 * 750
                plt.subplot(total // 2, 2, i + 1)
                y_min = np.min(sample) * 1.5 * 1000000
                y_max = np.max(sample) * 1.5 * 1000000
                y_min_label = int(y_min) // 10 * 10
                y_max_label = int(y_max) // 10 * 10
                plt.ylim(y_min, y_max)
                if y_min_label >= -80 and y_max_label <= 80:
                    plt.yticks([y_min_label, -50, 0, 50, y_max_label], [str(y_min_label), '-50', '0', '50', str(y_max_label)])
                elif y_min_label >= -80 and y_max_label > 80:
                    plt.yticks([y_min_label, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-50', '0', '50', '80', str(y_max_label)])
                elif y_min_label < -80 and y_max_label <= 80:
                    plt.yticks([y_min_label, -80, -50, 0, 50, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', str(y_max_label)])
                else:
                    plt.yticks([y_min_label, -80, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', '80', str(y_max_label)])
                plt.gca().invert_yaxis()
                plt.xlim(0, time_length // 2)
                data = sample * 1000000
                plt.xticks(x_ticks_point, x_ticks_label)
                plt.plot(data, scalex=False, scaley=False)
                plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')
                # 添加浅色的零线
                plt.axhline(0, color='lightgrey', linewidth=0.5)  # 添加y=0水平线
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (μV)')
                plt.title(f"{channle_names[i]}")
            plt.savefig(file_path)


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
            x_ticks_point = [i * 500 for i in np.arange(0, total_seconds + 0.5, 0.5)]
            x_ticks_label = [str(i) for i in np.arange(0, total_seconds + 0.5, 0.5)]
            for i, sample in enumerate(samples):
                plt.subplot(total // 2, 2, i + 1)
                y_min = sample.min * 1.5 * 1000000
                y_max = sample.max * 1.5 * 1000000
                y_min_label = int(y_min) // 10 * 10
                y_max_label = int(y_max) // 10 * 10
                plt.ylim(y_min, y_max)
                if y_min_label >= -80 and y_max_label <= 80:
                    plt.yticks([y_min_label, -50, 0, 50, y_max_label], [str(y_min_label), '-50', '0', '50', str(y_max_label)])
                elif y_min_label >= -80 and y_max_label > 80:
                    plt.yticks([y_min_label, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-50', '0', '50', '80', str(y_max_label)])
                elif y_min_label < -80 and y_max_label <= 80:
                    plt.yticks([y_min_label, -80, -50, 0, 50, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', str(y_max_label)])
                else:
                    plt.yticks([y_min_label, -80, -50, 0, 50, 80, y_max_label], [str(y_min_label), '-80', '-50', '0', '50', '80', str(y_max_label)])
                plt.gca().invert_yaxis()
                plt.xlim(0, sample.length)
                data = sample.sample * 1000000
                plt.xticks(x_ticks_point, x_ticks_label)
                plt.plot(data, scalex=False, scaley=False)
                plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')
                # 添加浅色的零线
                plt.axhline(0, color='lightgrey', linewidth=0.5)  # 添加y=0水平线
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude (μV)')
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
        res = SampleSeq(sample.event_idx, time_length, new_data, channel, mid-step, sample.type)
        # print(res.mid_idx - sample.mid_idx)
        # print(sample.start_idx - res.start_idx)
        # print(res.end_idx - sample.end_idx)
        return res
    
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