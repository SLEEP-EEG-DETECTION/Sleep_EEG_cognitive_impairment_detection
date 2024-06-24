from BaseSampler import BaseSampler
from SampleSequence import SampleSeq, TimeSeq
from BaseNormalizer import BaseNormalizer
from typing import List
import numpy as np

class Report:

    def __init__(self, k: int, event_count: int, name: str) -> None:
        """
        统计负采样情况
        ------------------------------------------------
        parameters:
            k： 采样比例
            event_count: 正样本数量
            name: edf名称
        ------------------------------------------------
        """
        self._k = k
        self._event_count = event_count
        self._dic = dict()
        self._name = name
    
    def add(self, succent_count: int) -> None:
        if succent_count in self._dic:
            self._dic[succent_count] += 1
        else:
            self._dic[succent_count] = 1
    
    def show(self) -> str:
        res = f"edf名字：{self._name} : \n"
        res += "\t正样本数量：{}\n".format(self._event_count)
        keys = self._dic.keys()
        t = sorted(keys)
        res += "\t负样本统计：\n"
        for k in t:
            res += "\t采样得到 {} 个样本：{}\n".format(k, self._dic[k])
        res += "\n\n"
        return res


class NegativeSamplerWithNormalize(BaseSampler):

    def __init__(self, event_idx_list: TimeSeq, time_length: int, k: int, min_time_distance: int, max_time_distance: int, name: str) -> None:
        """
        负采样器构造器
        Parameters:
        ---------------
        event_idx_list: 正样本采样时间序列点（横坐标），一维向量
        time_length: 采样时间长度，单位毫秒
        k: 采样比例，1个正样本附近采样k个负样本
        min_distance: 最小距离，单位毫秒
        max_distance： 最大距离，单位毫秒
        name: edf名字
        """
        super().__init__(event_idx_list, time_length, SampleSeq.NEGTIVE)
        self._min_distance = min_time_distance // 2
        self._max_distance = max_time_distance // 2
        self._sample_point_list = list()
        self._k = k
        self.__dic = dict()
        self._report = Report(k, len(self._event_idx_list), name)
    
    def __sample_points(self, channels: List[object]) -> None:
        n = self._k // 2
        m = self._k - n
        half_points_count = self._time_length // 4
        for i, index in enumerate(self._event_idx_list):
            # 左侧采样n个
            left_edge = self._time_length // 4
            total_success_count = 0  # 记录每个正样本采样得到的负样本数
            if i == 0:
                if left_edge < self._event_idx_list[i]:
                    start = max(left_edge, index - self._max_distance)
                    end = max(start, index - self._min_distance)
                    chs = self._generate_random(n, start, end)
                    succent_count = self._normalize(chs, channels, start, end)
                    total_success_count += succent_count
                    
            else:
                left_edge = self._event_idx_list[i - 1]
                start = max(left_edge + half_points_count, index - self._max_distance)
                end = max(start, index - self._min_distance)
                chs = self._generate_random(n, start, end)
                succent_count = self._normalize(chs, channels, start, end)
                total_success_count += succent_count
            
            # 右边采样m个
            right_edge = channels[0].length - (self._time_length // 4)
            if i < len(self._event_idx_list) - 1:
                right_edge = self._event_idx_list[i + 1]
            end = min(index + self._max_distance, right_edge)
            start = min(index + self._min_distance, end)
            chs = self._generate_random(m, start, end)
            succent_count = self._normalize(chs, channels, start, end)
            total_success_count += succent_count

            self._report.add(total_success_count)
        

    def _normalize(self, points: List[int], channels: List[object], left_edge: int, right_edge: int) -> int:
        dic = dict()
        for p in points:
            if p in self.__dic:
                continue
            self.__dic[p] = p
            ch_0 = channels[0]
            sample = self._normalize_one(p, ch_0, left_edge, right_edge)
            if sample is None:
                continue
            id = sample.mid_idx
            if id in dic:
                continue
            dic[id] = id
            self._sample_point_list.append(id)
        return len(dic)

    
    def _normalize_one(self, point: int, channel: object, left_edge: int, right_edge: int) -> SampleSeq | None:
        sample = self._sample_with_event(point, channel) # type: SampleSeq
        try_count = 20
        normalizer = BaseNormalizer()
        while try_count >= 0:
            sample = normalizer.normalize(sample)
            if sample.mid_idx <= left_edge or sample.mid_idx >= right_edge:
                return None
            if abs(sample.mid_idx - ((sample.arg_max_idx + sample.arg_min_idx) // 2)) < 4:
                return sample
            try_count -= 1 
        print("负采样失败")
        return None


    def _generate_random(self, count: int, start: int, stop: int) -> List[int]:
        numbers = np.arange(start=start, stop=stop, dtype=np.int32)
        return np.random.choice(numbers, size=count, replace=False).tolist() if numbers.__len__() > 0 else []
    
    def show(self) -> str:
        return self._report.show()
    
    def sample(self, channel: object) -> List[SampleSeq]:
        """
        单个通道采样
        """
        self.__sample_points([channel])
        return [self._sample_with_event(idx) for idx in self._sample_point_list]
    
    def sample_multi_channel(self, channels: List, channel_first: bool = True) -> List[List[SampleSeq]]:
        """
        多个通道采样
        Parameters:
        --------------
        channels: 通道数据
        channel_first: 是否通道数为第一维度, 为True时, 第一维时通道, 第二维是样本数, 为False时，第一维是样本数量, 第二维是通道
        """
        self.__sample_points(channels)
        res = []
        for channel in channels:
            res.append([self._sample_with_event(idx, channel) for idx in self._sample_point_list])
        if channel_first:
            return res
        return self.transpose_matrix(res)
    
 