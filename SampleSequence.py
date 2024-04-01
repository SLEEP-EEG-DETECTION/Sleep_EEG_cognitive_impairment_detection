import numpy as np
from typing import List
import matplotlib.pyplot as plt

TimeSeq = np.ndarray[int, np.dtypes.Int64DType] | List[int]


class SampleSeq(object):

    POSTIVE = "postive"

    NEGTIVE = "negtive"

    def __init__(self, event_idx: int, time_length: int, sample: np.ndarray[int, np.dtypes.Float64DType], channel: object, bias: int, type: str= "postive") -> None:
        """
        Parameters:
        --------------------------------
        event_idx: 标注值
        time_length: 时间长度，单位ms
        sample: 采样得到的序列
        channel: 被采样的序列
        bias: 时间坐标序列的偏差值
        type: 正样本还是负样本
        """
        self.__sample = sample
        self.__event_idx = event_idx
        self.__time_length = time_length
        self.__source = channel
        self.__type = self.POSTIVE
        self.__bias = bias
        self.__arg_max = np.argmax(sample, axis=0)
        self.__arg_min = np.argmin(sample, axis=0)
        self.__max = np.max(sample, axis=0)
        self.__min = np.min(sample, axis=0)
        self.__mean = np.mean(sample, axis=0)

    @property
    def start_idx(self) -> int:
        return self.__bias
    
    @property
    def end_idx(self) -> int:
        return self.__bias + self.time_length // 2

    @property
    def sample(self) -> np.ndarray[int, np.dtypes.Float64DType]:
        return self.__sample
    
    @property
    def channel_name(self) -> str:
        return self.__source.name
    
    @property
    def type(self) -> str:
        return self.__type
    
    @property
    def event_idx(self) -> int:
        return self.__event_idx
    
    @property
    def time_length(self) -> int:
        return self.__time_length
    
    @property
    def channel(self) -> object:
        return self.__source
    
    @property
    def arg_max(self) -> int:
        """
        采样序列的最大值坐标（在采样序列的坐标系下）
        """
        return self.__arg_max
    
    @property
    def arg_max_idx(self) -> int:
        """
        采样序列的最大值坐标（在原序列的坐标系下）
        """
        return self.__arg_max + self.__bias
    
    @property
    def arg_min(self) -> int:
        """
        采样序列的最小值坐标（在采样序列的坐标系下）
        """
        return self.__arg_min
    
    @property
    def arg_min_idx(self) -> int:
        """
        采样序列的最小值坐标（在原序列的坐标系下）
        """
        return self.__arg_min + self.__bias
    
    @property
    def max(self) -> float:
        """
        采样序列的最大值
        """
        return self.__max
    
    @property
    def min(self) -> float:
        """
        采样序列最小值
        """
        return self.__min
    
    @property
    def mean(self) -> float:
        return self.__mean
    
    @property
    def mid_idx(self) -> float:
        """
        采样序列中点坐标在原序列中的坐标
        """
        return (self.start_idx + self.end_idx) // 2
    
    def normalize(self, normalizer: object) -> 'SampleSeq':
        return normalizer.normalize(self)
    
    def plot(self, name: str = "original") -> None:
        self.__plot(name)
        plt.show(block=False)
    
    def __plot(self, name: str = "original") -> None:
        # plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
        # plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

        plt.plot(self.sample)
        plt.title(f"{self.channel_name} : {name}")

        x_min, y_min, x_max, y_max = (self.arg_min, self.sample[self.arg_min], self.arg_max, self.sample[self.arg_max])
        show_min= f"[{x_min} {y_min}]" 
        show_max= f"[{x_max} {y_max}]" 
        plt.plot(x_min, y_min, "ko")
        plt.plot(x_max, y_max, "ko")
        plt.annotate(show_min, xy=(x_min, y_min), xytext=(x_min, y_min))
        plt.annotate(show_max, xy=(x_max, y_max), xytext=(x_max, y_max))
    
    def export_plot(self, save_path: str) -> None:
        self.__plot()
        plt.savefig(save_path)

