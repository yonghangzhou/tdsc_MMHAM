#!/usr/bin/env python
# -*- coding:utf-8 -*- 


import pandas as pd
import pickle


def find_top_N(data, N):
    data_ = pd.Series(data)
    data_sorted = data_.sort_values(ascending=False)
    return data_sorted[:N].values,data_sorted.index.values

def get_values(data,index):
    data_ = pd.Series(data)
    return data_.loc[index]


def save_result(path,data):
    with open(path,'wb') as f:
        pickle.dump(data, f, 0)