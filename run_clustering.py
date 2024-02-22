
# coding: utf-8
# imports
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns;
import numpy as np
from sklearn.cluster import KMeans
import os
import debugpy
import tqdm
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sns.set()
import torch
import pickle
from datasets import Dataset, load_from_disk
from transformers import BertForMaskedLM
from CGMFormer import EmbExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# debugpy.listen(("192.168.72.58", 5681))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint')
    parser.add_argument('--data_path', type=str, help='Path to the data')
    parser.add_argument('--save_path', type=str, help='Path to save embeddings')
    parser.add_argument('--filter_target', type=str, help='Filter target')
    return parser.parse_args()

def main():
    args = parse_args()
    filter_target = "type"
    embex = EmbExtractor(model_type="SampleClassifier",
                         max_length=289,
                         num_classes=0,
                         emb_mode="sample",
                         sample_emb_style="mean_pool",
                         filter_data=None,
                         filter_target=args.filter_target,
                         max_nsamples=2000,
                         emb_layer=0,
                         emb_label=['index', 'p_id', 'id', 'filled', 'types', 'hba1c'],
                         labels_to_plot=['index'],
                         forward_batch_size=48,
                         nproc=16)

    embs = embex.extract_embs(args.checkpoint_path,  
                              args.data_path,
                              args.save_path,
                              f"mean_preTrainCheckpoint_Colas_1028_vec_{args.filter_target}")

if __name__ == "__main__":
    main()
