from SampleSequence import SampleSeq
from BaseSampler import BaseSampler

class BaseNormalizer(object):

    def normalize(self, sample_seq: SampleSeq) -> SampleSeq:
        mid = [(sample_seq.arg_max_idx + sample_seq.arg_min_idx) // 2]
        return BaseSampler(mid, sample_seq.time_length).sample(sample_seq.channel)[0]
        