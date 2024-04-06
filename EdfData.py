import mne
from mne.io.edf.edf import RawEDF
import numpy as np
from typing import List
from BaseSampler import BaseSampler
from SampleSequence import SampleSeq, TimeSeq
import os

class Channel(object):
    """"
    用于保存每个通道的值
    """

    def __init__(self, channel: np.ndarray[int, np.dtypes.Float64DType], name: str) -> None:
        assert channel.ndim == 1
        self.__data = channel
        self.__sample_data_list = []
        self.__name = name

    @property
    def length(self) -> int:
        return len(self.__data)
    
    @property
    def data(self) -> np.ndarray[int, np.dtypes.Float64DType]:
        return self.__data
    
    @property
    def sample_data_list(self) -> List[SampleSeq]:
        return self.__sample_data_list
    
    @property
    def name(self) -> str:
        return self.__name
    

class Edf(object):
    """
    用于读取edf数据，并获取各个通道的值以及k复合波标注
    """

    def __init__(self, path: str) -> None:
        self.__raw = mne.io.read_raw_edf(path)  # type: RawEDF
        tmp = self.__raw.get_data(picks=['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1']) # , 'EEG O1-A2', 'EEG O2-A1'])  # type: np.ndarray[int, np.dtypes.Float64DType]
        names = ["F3", "F4", "C3", "C4", "O1", "O2"]
        self.__channels_data = [Channel(tmp[i], names[i]) for i in range(0, len(tmp))]  # type: List[Channel]
        time_data, _ = mne.events_from_annotations(self.__raw, event_id={"K-复合波": 1})
        self.__k_complex_time = time_data[:, 0] # type: np.ndarray[int, np.dtypes.Int64DType]
        self.__name = os.path.splitext(os.path.basename(path))[0]
    
    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def raw(self) -> RawEDF:
        return self.__raw
    
    @property
    def channel_F3(self) -> Channel:
        """
        shape=(n,)
        """
        return self.__channels_data[0]

    @property
    def channel_F4(self) -> Channel:
        """
        shape=(n,)
        """
        return self.__channels_data[1]

    @property
    def channel_C3(self) -> Channel:
        """
        shape=(n,)
        """
        return self.__channels_data[2]
    
    @property
    def channel_C4(self) -> Channel:
        """
        shape=(n,)
        """
        return self.__channels_data[3]
    
    @property
    def channel_O1(self) -> Channel:
        """
        shape=(n,)
        """
        return self.__channels_data[4]
    
    @property
    def channel_O2(self) -> Channel:
        """
        shape=(n,)
        """
        return self.__channels_data[5]

    @property
    def all_channels_data(self) -> List[Channel]:
        """
        所有通道的数据
        shape=(6, n)
        """
        return self.__channels_data
    
    @property
    def k_complex_time(self) -> np.ndarray[int, np.dtypes.Int64DType]:
        """
        k复合波的标注时间点
        shape=(n,)
        """
        return self.__k_complex_time
    
