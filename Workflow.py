from EdfData import Edf, Channel
from SampleSequence import SampleSeq
from BaseNormalizer import BaseNormalizer
from BaseSampler import BaseSampler
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os

class Workflow(object):

    def __init__(self, edf: Edf) -> None:
        self.__edf = edf
        self.__sampler = None
        self.__normalizer = None

    @property
    def edf_data(self) -> Edf:
        return self.__edf
    
    @property
    def sampler(self) -> BaseSampler | None:
        return self.__sampler
    
    
    def set_sampler(self, value: BaseSampler) -> 'Workflow':
        self.__sampler = value
        return self

    @property
    def normalizer(self) -> BaseNormalizer | None:
        return self.normalizer

    def set_normalizer(self, value: BaseNormalizer):
        self.__normalizer = value
        return self

    def export_sample(self, save_dir: str) -> None:
        channels = self.__edf.all_channels_data
        if self.__sampler is None:
            print("采样器为空")
            return
        
        for channel in channels:
            save_chanel_path = os.path.join(save_dir, self.__edf.name ,channel.name)
            if not self.__check_dir(save_chanel_path):
                print("创建文件夹异常")
                raise Exception("创建文件夹异常")
            samples = self.__sampler.sample(channel)
            if self.__normalizer is not None:
                samples = [self.__normalizer.normalize(sample) for sample in samples]
            for i, sample in enumerate(samples):
                file_path = os.path.join(save_chanel_path, f"{i}.jpg")
                sample.export_plot(file_path)

    def export_all_channel_sample(self, save_dir: str) -> None:
        channels = self.__edf.all_channels_data
        if self.__sampler is None:
            print("采样器为空")
            return
        save_path = os.path.join(save_dir, self.__edf.name)
        self.__check_dir(save_path)
        sample_list = self.__sampler.sample_multi_channel(channels, False)  # type: List[List[SampleSeq]]
        for i, samples in enumerate(sample_list):
            file_path = os.path.join(save_path, f"{self.edf_data.name}-{samples[0].mid_idx}.jpg")
            if self.__normalizer is not None:
               samples = self.__normalizer.normalize_list_with_same_mid(samples)
            total = len(samples)
            plt.figure(figsize=(12, 10))
            for j, sample in enumerate(samples):
                # sample.show_plot()
                plt.subplot(total // 2, 2, j + 1)
                plt.ylim(-5*1e-5, 5*1e-5)
                plt.xlim(0, 2000)
                plt.xticks([0, 500, 1000, 1500, 2000], ['0', '1', '2', '3', '4'])
                plt.plot(sample.sample, scalex=False, scaley=False)
                plt.title(f"{sample.channel_name}-{j + 1}")
                # sample.sub_plot((total, 1, j + 1))
            plt.savefig(file_path)


    def __check_dir(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            return True
        except:
            return False
