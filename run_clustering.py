'''
@Author: liangzhongming
@Date: 
LastEditors: Please set LastEditors
LastEditTime: 2024-01-29 19:20:27
@Description: 请填写简介
'''
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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sns.set()
import torch
import pickle
from datasets import Dataset, load_from_disk
from transformers import BertForMaskedLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# debugpy.listen(("192.168.72.58", 5681))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

from CGMFormer import EmbExtractor
# initiate EmbExtractor
filter_target = "type"
# filter_target = "type" # zhao datasets
max_length = 289
embex = EmbExtractor(model_type="SampleClassifier",
                     max_length=max_length,
                     num_classes=0,
                     emb_mode="sample",
                     sample_emb_style="mean_pool",
                     filter_data=None,
                     filter_target=filter_target,
                     max_nsamples=2000,
                     emb_layer=0, #             -1: 2nd to last layer  0: last layer
                    #  emb_label=['id', 'types', 'age', 'bmi', 'hba1c', 'homa-b', 'homa-is', 'index', 'Fast_s', 'Fast_e', 'Dawn_s', 'Dawn_e', 'Breakfast_s', 'Breakfast_e', 'Lunch_s', 'Lunch_e', 'Dinner_s', 'Dinner_e'],
                    #  emb_label=['index', 'id', 'types', 'cerebrovascular_disease', 'coronary_heart_disease', 'peripheral_arterial_disease', 'nephropathy', 'neuropathy', 'retinopathy', 'macrovascular ', 'microvascular', 'complication'],
                     emb_label=['index', 'p_id', 'id', 'filled', 'types', 'hba1c'],
                     
                     labels_to_plot=['index'],
                    #  labels_to_plot=['types'],
                     forward_batch_size=48,
                     nproc=16)


# checkpoint_path_old288_sz1_ep3000 = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230806_083725_Shanghai_train_old288_m3060_sz1_L8_emb256_SL288_E3000_B12_LR2e-05_LSlinear_WU5000_Oadamw_DS2/checkpoint-87000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_010530_Fre3060_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # 3060
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_003104_TFIDF4560_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # TF-IDF
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_061610_Fre4570_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # 4570

#288 + 1
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# # checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# total_data = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_total"
# # /save_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/288_emb/heatmap/"
# save_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/288_emb/"

# # T1D
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# Andreson = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Andreson"
# Sence = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/SENCE"

# CITY
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
CITY_interpolatedData = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_interpolated"
CITY_origin = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_origin"


# Colas
Colas = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas"

# 1028_Colas
Colas_1028 = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas_1028_hfds"
Colas_1028_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231028_125607_dim128_linear_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-546"

# Excercise
ShanghaiExercise = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/10_5_Excercise_total"
checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"

# Diet
ShanghaiDiet = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/10_14_ShanghaiDiet_total"
checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"


# 1017_Zhao
# Zhao_1017 = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/1017_Zhao_total"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"

# # 96 + 1
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
# total_data = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downsampled_Shanghai_total_96"
save_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/"

# # Zhao 96
# total_data = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Zhao_total"

embs = embex.extract_embs(Colas_1028_checkpoint_path,  
                          Colas_1028,
                          save_path,
                          f"mean_preTrainCheckpoint_Colas_1028_vec_{filter_target}")

# embs = embex.extract_embs(checkpoint_path,
#                           total_data,
#                           save_path,
#                           "vec_total_823")

# plot_style = "umap"
# embex.plot_embs(embs=embs, 
#                 plot_style=plot_style,
#                 output_directory="/share/home/liangzhongming/930/CGMformer/figures/",  
#                 output_prefix=f"CLS_Andreson_emb_{plot_style}_plot_831")