import os
from typing import List

import numpy as np
from FileUtils import FileUtils
from EdfData import Edf
from BaseSampler import BaseSampler
from NegativeSamplerWithNormalize import NegativeSamplerWithNormalize
from SampleSequence import SampleSeq
import pickle

class LabelData:

    def __init__(self, path: str) -> None:
        self._edf_name = os.path.basename(os.path.dirname(path)) + ".edf"
        self._center = int(os.path.splitext(os.path.basename(path))[0].split('-')[1])
    
    @property
    def edf_name(self) -> str:
        return self._edf_name
    
    @property
    def center(self) -> int:
        return self._center

class GenerateData:

    def __init__(self, label_path: str, edf_path: str) -> None:
        lable_list = FileUtils.get_file_list(label_path, ".jpg")
        self._label_data_list = [LabelData(x) for x in lable_list]
        self._edf_dic = FileUtils.get_file_dic(edf_path, ".edf")
    
    def generate(self, save_path: str) -> bool:
        postitive_label_dic = dict()
        for label_data in self._label_data_list: 
            if label_data.edf_name not in postitive_label_dic:
                postitive_label_dic[label_data.edf_name] = list()
            postitive_label_dic[label_data.edf_name].append(label_data.center)

        
        post_data = list()
        negative_data = list()
        for k, v in  postitive_label_dic.items():
            edf = Edf(self._edf_dic[k])
            sampler = BaseSampler(v, 1500)
            negative_sampler = NegativeSamplerWithNormalize(v, 1500, 10, 750, 30000, edf.name)
            x = sampler.sample_multi_channel(edf.all_channels_data, False)
            y = negative_sampler.sample_multi_channel(edf.all_channels_data, False)
            print(negative_sampler.show())
            post_data.extend(x)
            if y is not None:
                negative_data.extend(y)
        
        postitive_save_path = os.path.join(save_path, "postitive.pkl")
        negative_save_path = os.path.join(save_path, "negative.pkl")
        self._convert(postitive_save_path, post_data)
        self._convert(negative_save_path, negative_data)
        
    
    def _convert(self, save_path: str, data: List[List[SampleSeq]]) -> None:
        for j in range(len(data)):
            for i in range(len(data[j])):
                data[j][i] = data[j][i].sample
            data[j] = np.array(data[j])
        
        np_data = np.array(data)
        with open(save_path, 'wb') as f:
            pickle.dump(np_data, f)
        
            
g = GenerateData("./LabelData_1", "./edf")
g.generate("./")

