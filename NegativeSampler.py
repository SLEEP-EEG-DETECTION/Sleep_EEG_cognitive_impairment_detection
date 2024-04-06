from BaseSampler import BaseSampler
from SampleSequence import SampleSeq, TimeSeq
from typing import List
import numpy as np

class NegativeSampler(BaseSampler):

    def __init__(self, event_idx_list: TimeSeq, time_length: int, k: int, min_distance: int, max_distance: int) -> None:
        super().__init__(event_idx_list, time_length, SampleSeq.NEGTIVE)
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._sample_point_list = []
        self._k = k

    
    def __init_sample_points(self, k: int, max_index: int) -> None:
        n = k // 2
        m = k - n
        for i, index in enumerate(self._event_idx_list):
            # 左边采样n个
            left_edge = 375 # 保证边界采样后不会少于750个点
            if i == 0:
                if left_edge < self._event_idx_list[i]:
                    start = max(left_edge, index - self._max_distance)
                    end = max(start, index - self._min_distance)
                    self._sample_point_list.extend(self._generate_random(n, start, end))
            elif i > 0:
                left_edge = self._event_idx_list[i - 1]
                start = max(left_edge, index - self._max_distance)
                end = max(start, index - self._min_distance)
                self._sample_point_list.extend(self._generate_random(n, start, end))

            # 右边采样m个
            right_edge = max_index
            if i < len(self._event_idx_list) - 1:
                right_edge = self._event_idx_list[i + 1]
            end = min(index + self._max_distance, right_edge)
            start = min(index + self._min_distance, end)
            self._sample_point_list.extend(self._generate_random(m, start, end))
            
    
    def _generate_random(self, count: int, start: int, stop: int) -> List[int]:
        numbers = np.arange(start=start, stop=stop, dtype=np.int64)
        return np.random.choice(numbers, size=count, replace=False).tolist() if numbers.__len__() > 0 else []


    def sample(self, channel: object) -> List[SampleSeq]:
        if len(self._sample_point_list) == 0:
            self.__init_sample_points(self._k, channel.length - 375) # 保证边界采样不会少于750个点
        return [self._sample_with_event(index, channel) for index in self._sample_point_list]

    def sample_multi_channel(self, channels: List) -> List[List]:
        return [self.sample(channel) for channel in channels]