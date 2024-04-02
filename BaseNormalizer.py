from SampleSequence import SampleSeq
from BaseSampler import BaseSampler
from typing import List

class BaseNormalizer(object):

    def normalize(self, sample_seq: SampleSeq) -> SampleSeq:
        mid = [(sample_seq.arg_max_idx + sample_seq.arg_min_idx) // 2]
        return BaseSampler(mid, sample_seq.time_length).sample(sample_seq.channel)[0]
    
    def normalize_list_with_same_mid(self, sample_seq_list: List[SampleSeq]) -> List[SampleSeq]:
        mid = (sample_seq_list[0].arg_max_idx + sample_seq_list[0].arg_min_idx) // 2
        sampler = BaseSampler([mid], sample_seq_list[0].time_length)
        return [sampler.sample(it.channel)[0] for it in sample_seq_list]
        