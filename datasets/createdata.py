import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call
import torch
import torch.nn as nn
import random

# from src.create_dataset import mosi_test


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def create(data):
    new_data=[]
    for i in range(len(data)):
        sample=data[i]
        label=sample[1]
        label=label.split(',')
        if len(label)>1: label=label[1]
        else : label=label[0]
        new_label = str(int(np.round(float(label))))

        for i in range(3):
            new_label=new_label+','+new_label
        new_data.append((sample[0],new_label,sample[2],label))
        # print(sample)
    return new_data

mosei_dev,mosei_train,mosei_test,mosi_dev,mosi_train,mosi_test=(
    load_pickle("mosei_dev.pkl"),load_pickle("mosei_train.pkl"),load_pickle("mosei_test.pkl"),load_pickle("mosi_dev.pkl"),
    load_pickle("mosi_train.pkl"),load_pickle("mosi_test.pkl"))
# print(mosei_dev)
mosei_dev,mosei_train,mosei_test,mosi_dev,mosi_train,mosi_test=(
    create(mosei_dev),create(mosei_train),create(mosei_test),create(mosi_dev),create(mosi_train),create(mosi_test)
)

mos_train=mosei_train+mosi_train
mos_dev=mosei_dev+mosi_dev

random.shuffle(mos_train)
random.shuffle(mos_dev)

to_pickle(mos_train,"./MOS/mos_train.pkl")
to_pickle(mos_dev,"./MOS/mos_dev.pkl")
to_pickle(mosei_test,"./MOS/new_mosei_test.pkl")
to_pickle(mosi_test,"./MOS/new_mosi_test.pkl")


print(mosei_dev)