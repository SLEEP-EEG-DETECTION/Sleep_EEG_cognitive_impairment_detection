from EdfData import Edf
from FileUtils import FileUtils
from BaseSampler import BaseSampler
from Utils import Utils
from Workflow import Workflow
from BaseNormalizer import BaseNormalizer

edf = Edf("./edf/001.edf")
# event_list = [1843804]
# sample = BaseSampler(event_list, 4000)
# res = sample.sample_multi_channel(edf.all_channels_data, channel_first=False)
# print(f"中点坐标: {res[0][0].mid_idx}")
# Utils.plot_multi_channel(res[0], 4000)
# x = 0

work = Workflow(edf)
event_list = [1843804, 1870295, 1875774]
sample = BaseSampler(event_list, 4000)
nor = BaseNormalizer()
res = sample.sample_multi_channel(edf.all_channels_data, channel_first=False)
x = [nor.normalize_list_with_same_mid(y) for y in res]
work.export_all_channel_sample_from_sampleSeqs(res, "001", "./tmp")
work.export_all_channel_sample_from_sampleSeqs(x, "001", "./tmp2")