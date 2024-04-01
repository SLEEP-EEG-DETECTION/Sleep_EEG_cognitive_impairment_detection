from SampleSequence import SampleSeq, TimeSeq
from typing import List

class BaseSampler(object):

    def __init__(self, event_idx_list: TimeSeq, time_length: int, type: str = SampleSeq.POSTIVE) -> None:
        """
        根据单时间点序列进行采样
        Parameters:
        -----------
        event_time: 采样的时间点序列
        time_length: 采样时间长度，单位毫秒
        data: 采样的数据源

        Return
        ----------
        根据单时间点序列采样得到的序列数组
        """
        self._type = type
        self._event_idx_list = event_idx_list
        self._time_length = time_length
   
    def _sample_with_event(self, event_idx, channel: object) -> SampleSeq:
        """
        根据单一时间点进行采样
        Parameters:
        -----------
        event_time: 采样的时间点
        time_length: 采样时间长度，单位毫秒
        data: 采样的数据源

        Return
        ----------
        采样得到的序列
        """
        step = self._time_length // 4  # 每个点2毫秒，再对半区间，得到中心点左右的采样点数
        l = max(event_idx - step, 0)  # 防止左侧越界
        r = min(event_idx + step, len(channel.data))  # 防止右侧越界
        length = r - l  # 计算实际的采样点数，乘2就到的采样的时间长度，单位毫秒
        sample_data = channel.data[l:r]
        return SampleSeq(event_idx, length * 2, sample_data, channel, l, self._type)

    def sample(self, channel: object) -> List[SampleSeq]:
        return [self._sample_with_event(it, channel) for it in self._event_idx_list]
    
    def sample_multi_channel(self, channels: list) -> List[list]:
        return [self.sample(channel) for channel in channels]