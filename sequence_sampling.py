import datetime
import matplotlib as plt

import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

import tqdm
import random


from datasets import Dataset
import pandas as pd
import numpy as np
import pytz
import torch
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset

from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf, adfuller
from scipy.signal import welch
from scipy.stats import chisquare


def sliding_window_sampling(data, window_size, step_size):
    samples = []
    for i in range(0, len(data) - window_size + 1, step_size):
        sample = data[i:i + window_size]
        samples.append(sample)
    return samples

def random_sampling_from_intervals(data, num_intervals, num_samples):
    interval_size = len(data) // num_intervals
    samples = []
    for _ in range(num_samples):
        sample = []
        for i in range(num_intervals):
            start = i * interval_size
            end = (i + 1) * interval_size
            point = random.choice(data[start:end])
            sample.append(point)
        samples.append(sample)
    return samples

def interval_sampling(sample_path, save_path):
    train_dataset = load_from_disk(sample_path)
    num_intervals = 96
    num_samples = 10

    sampled_input_ids = []
    sampled_types = []

    for input_ids, type_ in zip(train_dataset['input_ids'], train_dataset['types']):
        sampled_ids = random_sampling_from_intervals(input_ids, num_intervals, num_samples)
        sampled_type_ = [type_] * len(sampled_ids)
        sampled_input_ids.extend(sampled_ids)
        sampled_types.extend(sampled_type_)

    sampled_dataset = Dataset.from_dict({
        'input_ids': sampled_input_ids,
        'types': sampled_types
    })
    
    sampled_dataset.save_to_disk(save_path)


def uniform_fixed_sampling(data, window_size, num_sections):
    # Split data into sections
    sections = np.array_split(data, num_sections)

    # Initialize list to hold samples
    samples = []

    # For each position within a section
    for pos in range(window_size):
        # Sample the same position from each section
        sample = [section[pos] for section in sections]
        samples.append(sample)

    return samples

def samplingV3(sample_path, save_path):
    train_dataset=load_from_disk(sample_path)
    # Determine the number of sections based on the window size
    original_length = len(train_dataset['input_ids'][0])  # assuming all sequences have the same length
    window_size = 3
    num_sections = original_length // window_size

    # Initialize lists to hold the sampled data
    sampled_input_ids = []
    sampled_types = []

    # Loop over the original data
    for input_ids, type_ in zip(train_dataset['input_ids'], train_dataset['types']):
        # # 处理为2分类问题是解开
        # if type_ == "2":
        #     type_ = "1"

        # Sample the input_ids sequence
        sampled_ids = uniform_fixed_sampling(input_ids, window_size, num_sections)
        
        # Repeat the type_ for the number of samples
        sampled_type_ = [type_] * len(sampled_ids)
        
        # Append to the lists
        sampled_input_ids.extend(sampled_ids)
        sampled_types.extend(sampled_type_)

    # Create a new dataset from the sampled data
    sampled_dataset = Dataset.from_dict({
        'input_ids': sampled_input_ids,
        # 'types': sampled_types.replace("2", "1"),
        'types': sampled_types
    })

    # Save the new dataset
    sampled_dataset.save_to_disk(save_path)



def sampling(sample_path, save_path):
    train_dataset=load_from_disk(sample_path)
    # Determine step size based on the number of required samples
    original_length = len(train_dataset['input_ids'][0])  # assuming all sequences have the same length
    num_samples = 10
    window_size = 96
    step_size = (original_length - window_size) // (num_samples - 1)

    # Initialize lists to hold the sampled data
    sampled_input_ids = []
    sampled_types = []

    # Loop over the original data
    for input_ids, type_ in zip(train_dataset['input_ids'], train_dataset['types']):
        # Sample the input_ids sequence
        sampled_ids = sliding_window_sampling(input_ids, window_size, step_size)
        
        # Repeat the type_ for the number of samples
        sampled_type_ = [type_] * len(sampled_ids)
        
        # Append to the lists
        sampled_input_ids.extend(sampled_ids)
        sampled_types.extend(sampled_type_)

    # Create a new dataset from the sampled data
    sampled_dataset = Dataset.from_dict({
        'input_ids': sampled_input_ids,
        'types': sampled_types
    })

    # Save the new dataset
    sampled_dataset.save_to_disk(save_path)


from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf, adfuller
from scipy.signal import welch
from scipy.stats import chisquare


def qualityAnalysis(sample_path, save_path):

    
    train_dataset = load_from_disk(sample_path)
    sampled_dataset = load_from_disk(save_path)

    # Calculate statistics for original data
    original_means = [np.mean(seq) for seq in train_dataset['input_ids']]
    original_vars = [np.var(seq) for seq in train_dataset['input_ids']]
    original_acfs = [acf(seq, nlags=50) for seq in train_dataset['input_ids']]
    original_p_values = [adfuller(seq)[1] for seq in train_dataset['input_ids']]
    original_psds = [welch(seq)[1] for seq in train_dataset['input_ids']]  # select PSDs

    # Calculate statistics for sampled data
    sampled_means = [np.mean(seq) for seq in sampled_dataset['input_ids']]
    sampled_vars = [np.var(seq) for seq in sampled_dataset['input_ids']]
    sampled_acfs = [acf(seq, nlags=50) for seq in sampled_dataset['input_ids']]
    sampled_p_values = [adfuller(seq)[1] for seq in sampled_dataset['input_ids']]
    sampled_psds = [welch(seq)[1] for seq in sampled_dataset['input_ids']]  # select PSDs

    # Compare distributions of means, vars, acfs and psds using Kolmogorov-Smirnov test
    ks_result_means = ks_2samp(original_means, sampled_means)
    ks_result_vars = ks_2samp(original_vars, sampled_vars)
    ks_result_acfs = ks_2samp(np.concatenate(original_acfs), np.concatenate(sampled_acfs))
    ks_result_psds = ks_2samp(np.concatenate(original_psds), np.concatenate(sampled_psds))

    # Print KS test results
    print("KS test result for means:", ks_result_means)
    print("KS test result for vars:", ks_result_vars)
    print("KS test result for ACFs:", ks_result_acfs)
    print("KS test result for PSDs:", ks_result_psds)



if __name__ == '__main__':

    sample_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_test"
    save_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_test_96"
    # V1
    # sampling(sample_path, save_path)

    # # V2
    # interval_sampling(sample_path, save_path)

    # V3
    samplingV3(sample_path, save_path)

    # qualityAnalysis(sample_path, save_path)



    