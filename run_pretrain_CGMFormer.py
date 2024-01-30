#!/usr/bin/env python
# coding: utf-8

# run with:
# deepspeed --num_gpus=2 run_pretrain_CGMFormer.py

import datetime
import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

import pickle
import random
import subprocess

import numpy as np
import pytz
import torch
from datasets import load_from_disk
# from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from transformers import BertConfig, TrainingArguments
from torch.utils.tensorboard import SummaryWriter
from CGMFormer import CGMFormerPretrainer, BertForMaskedLM
import debugpy

# For vscode remote debugging
# debugpy.listen(("192.168.72.59", 5681))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

seed_num = 42
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# set local time/directories
timezone = pytz.timezone("Asia/Shanghai")
rootdir = "/share/home/liangzhongming/930/CGMformer/output/output_ablation"


token_dict_path = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'

# # old 288
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_train"
# valid_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_valid"

# # old 288 96
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_train_subSampleV3"
# valid_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_valid_subSampleV3"

# 8_11 downsampling 96
train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downsampled_Shanghai_total_96"
valid_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/96/downsampled_CV_2_train_96"

# # 8_11 288
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_total"
# valid_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_2/train"

# 144
train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_7_data/Shanghai_downsampled_144"
# downsampling
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_1650"
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_1150"
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_750"
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_450"
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_250"


# Default Test
# train_datsset_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_originData"


# use_cls_token = True

# set model parameters
# model type
model_type = "bert"
# max input size
max_input_size = 97 # <cls> + 96/ <cls> + 288
# number of layers
num_layers = 4
# number of attention heads
num_attn_heads = 8 # 8——>4
# number of embedding dimensions
num_embed_dim = 128
# intermediate size
intermed_size = num_embed_dim * 4 # num_embed_dim * 4——>num_embed_dim * 2
# activation function
activ_fn = "gelu" # relu->gelu
# initializer range, layer norm, dropout
initializer_range = 0.02 # Bert default 0.02
layer_norm_eps = 1e-12 
attention_probs_dropout_prob = 0.02 # Bert default 0.1 # 0.02->0.1
hidden_dropout_prob = 0.02 # Bert default 0.1 # 0.02->0.1 

# set training parameters
# number gpus
num_gpus = 2
# batch size for training and eval
batch_size = 48 
# max learning rate
max_lr = 4e-4 # 8e-6(2000epoch) 4e-6 best
# learning schedule
lr_schedule_fn = "linear" # linear->cossin
# warmup steps
warmup_steps = 2000 # 2000->200 for 100sample
# number of epochs
epochs = 3000 
# optimizer
optimizer = "adamw"
# weight_decay
weight_decay = 0.001 # 0.001->0.01

# output directories
decs = "mask_97_bs48_TFIDF4560"
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"{datestamp}_{decs}_L{num_layers}_H{num_attn_heads}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")


# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")


# make training and model output directories
subprocess.call(f"mkdir {training_output_dir}", shell=True)
subprocess.call(f"mkdir {model_output_dir}", shell=True)

with open(token_dict_path, "rb") as fp:
    token_dictionary = pickle.load(fp)

# model configuration
config = {
    "hidden_size": num_embed_dim,
    "num_hidden_layers": num_layers,
    "initializer_range": initializer_range,
    "layer_norm_eps": layer_norm_eps,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "hidden_dropout_prob": hidden_dropout_prob,
    "intermediate_size": intermed_size,
    "hidden_act": activ_fn,
    "max_position_embeddings": max_input_size,
    "model_type": model_type,
    "num_attention_heads": num_attn_heads,
    "pad_token_id": token_dictionary.get("<pad>"),
    "vocab_size": len(token_dictionary),
}

config = BertConfig(**config)
model = BertForMaskedLM(config)
print(f"Number of parameters in the model: {model.num_parameters():,}")
model = model.train()

# define the training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "per_device_train_batch_size": batch_size,
    "num_train_epochs": epochs,
    "save_strategy": "steps",
    "evaluation_strategy": 'steps',
    "eval_steps": 1000,
    "save_steps": 1000, 
    "logging_steps": 100,
    "output_dir": training_output_dir,
    "logging_dir": logging_dir,
}
training_args = TrainingArguments(**training_args)

# 4 eval
def compute_metrics(pred):
    mlm_labels = pred.label_ids
    mlm_preds = pred.predictions.argmax(-1)
    mlm_acc = np.equal(mlm_preds, mlm_labels)
    mask = np.not_equal(mlm_labels, -100)
    mlm_acc = mlm_acc[mask]
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total

    return {'accuracy': mlm_acc} if mlm_total > 0 else {}

print("Starting training.")

# define the trainer
trainer = CGMFormerPretrainer(
    model=model,
    args=training_args,
    train_dataset=load_from_disk(train_datsset_path),
    eval_dataset=load_from_disk(valid_datsset_path),
    token_dictionary=token_dictionary,
    compute_metrics = compute_metrics,
)

# train
trainer.train()

# save model
trainer.save_model(model_output_dir)
