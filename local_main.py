from EdfData import Edf, Channel
from SampleSequence import SampleSeq
from BaseNormalizer import BaseNormalizer
from BaseSampler import BaseSampler
from NegativeSampler import NegativeSampler
from typing import List
import os
from Workflow import Workflow
import matplotlib.pyplot as plt
from Utils import Utils
import numpy as np
import glob
from tqdm import tqdm
import mne
import pickle

def main():
    path = "001.edf"
    edf = Edf(path)
    sampler = BaseSampler(edf.k_complex_time, 1500)
    normalizer = BaseNormalizer()
    workflow = Workflow(edf).set_sampler(sampler).set_normalizer(normalizer)
    workflow.export_sample("./postive")

def test():
    path = "001.edf"
    edf = Edf(path)
    sampler = BaseSampler(edf.k_complex_time[0:1], 6000)
    sample = sampler.sample(edf.channel_C3)[0]
    plt.subplot(2, 1, 1)
    sample.show_plot()
    # plt.show()
    new_ample = BaseNormalizer().normalize(sample)
    plt.subplot(2, 1, 2)
    new_ample.show_plot("normalized")
    # plt.show()
    tmp = 0

def export_all():
    path = "001.edf"
    edf = Edf(path, picks=['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1'])
    sampler = BaseSampler(edf.k_complex_time, 4000)
    normalizer = BaseNormalizer()
    workflow = Workflow(edf).set_sampler(sampler).set_normalizer(normalizer)
    workflow.export_all_channel_sample("LabelData/postive")

def export_standard_train_data(path="001.edf", save_dir="LabelData", standard_time_length=1500, negative_sample=False, picks=['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1'], move_no_k_path=r"G:\eef_edf\MCI_edf_no_k"):
    # 1500窗口的样本保存
    try:
        edf = Edf(path, picks=picks)
    except:
        print(f"##### Error file: {path} ! #####")
        # import shutil
        # shutil.move(path, move_no_k_path) # 没有k波的，移走
        return
    normalizer = BaseNormalizer()
    sampler = BaseSampler(edf.k_complex_time, standard_time_length)
    channels_sampler = sampler.sample_multi_channel([edf.channel_F3, edf.channel_F4, edf.channel_C3, edf.channel_C4], channel_first=False)
    new_samples = [normalizer.normalize_list_with_same_mid(tmp) for tmp in channels_sampler] # 获取标准化后的sample：new_ample[0][0].sample
    
    # 4000范围的样本画图，同1500样本的中心点 和 保存（一个病例文件）
    expanded_new_sample = [Utils.slice_multi_channel_samples(channels_sample, 4000) for channels_sample in new_samples]
    Workflow.export_all_channel_sample_from_sampleSeqs(expanded_new_sample, edf.name, save_dir)
    
    if negative_sample:
        # 负采样
        negative_sampler = NegativeSampler(np.array([sampleseq.mid_idx for sampleseq in new_samples[0]]), standard_time_length, k=10, min_distance=375, max_distance=1500)
        negative_channels_sampler = negative_sampler.sample_multi_channel([edf.channel_F3, edf.channel_F4, edf.channel_C3, edf.channel_C4])

    t = 0

def save_label_data_from_edf(source_path, save_path, negative_sample=False, picks=['EEG F3-A2', 'EEG F4-A1', 'EEG C3-A2', 'EEG C4-A1'], move_path=r"G:\eef_edf\MCI_edf_ori", move_no_k_path=r"G:\eef_edf\MCI_edf_no_k"):
    
    for file in tqdm(os.listdir(source_path)): 
        path = os.path.join(source_path, file)
        try:
            export_standard_train_data(path=path, save_dir=save_path, standard_time_length=1500, negative_sample=negative_sample, picks=picks, move_no_k_path=move_no_k_path)
            # import shutil
            # shutil.move(path, move_path) # 有k波的移走
        except:
            print(f"##### Error file: {file} ! #####")
            continue
    print("##### Finished! #####")


def trans_to_pickle(path):
    edf = Edf(path)
    sampler = BaseSampler(edf.k_complex_time, 1500)
    normalizer = BaseNormalizer()
    channels_sampler = sampler.sample_multi_channel([edf.channel_F3, edf.channel_F4, edf.channel_C3, edf.channel_C4], channel_first=False)
    new_samples = [normalizer.normalize_list_with_same_mid(tmp) for tmp in channels_sampler]
    all_data = Utils.export_sample(new_samples)
    dump_data = [np.array(all_data[i:i+2]) for i in range(0, len(all_data) // 2)]
    pickle.dump(dump_data, open(r'F:\中大-脑神经\data.pkl', 'wb'))
    

if __name__ == '__main__':
    
    picks_dict = {
        1: ['EEG F3-A2',  'EEG F4-A1',  'EEG C3-A2',  'EEG C4-A1'], # 001-200
        2: ['EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF'], # 201-268
        3: ['EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref', 'EEG C4-Ref'] # 三九
    }
    source_path = r"data_ori"  # 目录
    save_path   = r"LabelData" # 输出
    
    move_path   = r"H:\eef_edf\001-200 Preprocess\MCI_edf_ori" # 不用改
    move_no_k_path = r"H:\eef_edf\001-200 Preprocess\MCI_edf_no_k" # 不用改
    
    save_label_data_from_edf(source_path=source_path, save_path=save_path, negative_sample=False, picks=picks_dict[1], move_path=move_path, move_no_k_path=move_no_k_path)

    # trans_to_pickle(r"F:\中大-脑神经\Sleep_EEG_cognitive_impairment_detection\001.edf")
    # export_standard_train_data(path=r"G:\eef_edf\MCI_edf\053.edf")
    # print(plt.rcParams["figure.figsize"])
