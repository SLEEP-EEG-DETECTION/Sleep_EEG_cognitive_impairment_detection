from EdfData import Edf, Channel
from SampleSequence import SampleSeq
from BaseNormalizer import BaseNormalizer
from BaseSampler import BaseSampler
from typing import List
import os
from Workflow import Workflow
import matplotlib.pyplot as plt

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
    sampler = BaseSampler(edf.k_complex_time[0:1], 1500)
    sample = sampler.sample(edf.channel_C3)[0]
    plt.subplot(2, 1, 1)
    sample.plot()
    # plt.show()
    new_ample = BaseNormalizer().normalize(sample)
    plt.subplot(2, 1, 2)
    new_ample.plot("normalized")
    # plt.show()
    tmp = 0

test()
