from EdfData import Channel
from SampleSequence import SampleSeq
from typing import List
import matplotlib.pyplot as plt
import os


class Utils:

    @staticmethod
    def plot_multi_channel(samples: List[SampleSeq], time_length: int) -> None:
        """
        Parameters:
        ---------------
        samples: n个通道的数据,
        time_length: 要画图的时间长度
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
    def export_multi_channle_samples(save_dir: str, samples_list: List[List[SampleSeq]]) -> None:
        """
        导出图片
        Parameters:
        -----------------
        save_dir: 保存的目录
        samples_list: n_samples * m_channle 的数据，n个采样数据，每个采样数据有m个通道
        """
        if not Utils.check_dir(save_dir):
            raise Exception(f"目录{save_dir}不存在")
        for samples in samples_list: # 取出每个采样数据，采样数据包含多个通道的
            file_path = os.path.join(save_dir, f"{samples[0].mid_idx}.jpg")
            plt.figure(figsize=(12, 10))
            total = len(samples)
            total_seconds = samples[0].time_length // 1000
            x_ticks_point = [i * 500 for i in range(total_seconds + 1)]
            x_ticks_label = [str(i) for i in range(total_seconds + 1)]
            for i, sample in enumerate(samples):
                plt.subplot(total // 2, 2, i + 1)
                plt.ylim(-5*1e-5, 5*1e-5)
                plt.xlim(0, sample.length)
                plt.xticks(x_ticks_point, x_ticks_label)
                plt.plot(sample.sample, scalex=False, scaley=False)
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
    def slice_one_channel_sample(sample: SampleSeq, time_length: int) -> SampleSeq:
        """
        根据给定的数据的中心点，重新采样到指定的时间长度
        Parameters:
        ---------------
        sample: 1个通道的采样数据
        time_length: 时间长度

        Return
        ---------------
        返回重新采样后的数据
        """
        step = time_length // 4
        mid = sample.mid_idx
        channel = sample.channel  # type: Channel
        new_data = channel.data[mid-step: mid + step]
        return SampleSeq(sample.event_idx, time_length, new_data, channel, sample.bias, sample.type)
    
