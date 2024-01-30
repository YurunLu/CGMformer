'''
@Author: liangzhongming
@Date: 
LastEditors: Please set LastEditors
LastEditTime: 2024-01-29 14:43:37
@Description: 
'''
import os

GPU_NUMBER = [1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# imports
from collections import Counter
import datetime
import pickle
import torch
import random
import subprocess
import seaborn as sns
import numpy as np
import sys
import debugpy

sns.set()
from datasets import load_from_disk
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer
from transformers.training_args import TrainingArguments

from CGMFormer import BertForSequenceClassification
from CGMFormer import DataCollatorForCellClassification
from CGMFormer import ClasssifyTrainer


# debugpy.listen(("192.168.72.57", 5682))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

seed_num = 51
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 51
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dictionary = pickle.load(f)

# paths
# train_path = "/share/home/liangzhongming/930/downstream_cl_data/train"
# test_path = "/share/home/liangzhongming/930/downstream_cl_data/train/test"

# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_finetune"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_test"

# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_finetune"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_test"

# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230805_185023_Shanghai_train_1581_288_m3060_L8_emb256_SL288_E2000_B12_LR8e-06_LSlinear_WU4000_Oadamw_DS2/checkpoint-66000" # 2000epoch sz10
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230805_224807_Shanghai_train_1581_288_m3060_sz1_L8_emb256_SL288_E3000_B12_LR2e-05_LSlinear_WU5000_Oadamw_DS2/checkpoint-99000" # 新288 3000epoch sz1
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_010530_Fre3060_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # old288 3000epoch sz1
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_010530_Fre3060_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_003104_TFIDF4560_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230810_002350_Fre3060_SZ1_L8_emb256_SL96_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # 96 m3060
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230810_131518_Fre3060_SZ10_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # tokenizer10 m3060

# 80w
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"

# 50w
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230829_145239_TFIDF4560_sincos_SZ1_L4_H4_emb128_SL289_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_1/train"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_1/test"


# Colas
# train_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas_train"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas_test"


# Zhao treat as a single label
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_1/train"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_1/test"

# checkpoint
# classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-1248"
# total_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_total"
# colas_1028 = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas_1028_hfds"


# data
# 1917
checkpoint_1917_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231028_125607_dim128_linear_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# 1650
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231110_003406_volume_1650_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-27000"
# 1150
# checkpoint_1150_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231109_220859_volume_1150_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-18000"
# 750
# checkpoint_750_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231107_142822_volume_750_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-12000"
# 450
# checkpoint_450_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231110_073038_volume_450_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-9000"
# 250
# checkpoint_250_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231106_155407_volume_250_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-6000"

# mask
# checkpoint_mask_random_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231112_135558_mask_97_bs48_random4560_L4_H8_emb128_SL97_E3000_B24_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"

# dim
checkpoint_dim256_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231027_123202_dim256_L4_H8_emb256_SL289_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
checkpoint_dim128_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231027_164230_dim128_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
checkpoint_dim64_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231027_202529_dim64_L4_H8_emb64_SL289_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
checkpoint_dim32_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231028_000917_dim32_L4_H8_emb32_SL289_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"

# 96
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
checkpoint_l97_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downsampled_CV_1_train_96"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downsampled_CV_1_test_96"
train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_1/train"
test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_1/test"
# checkpoint_97_linear_path = '/share/home/liangzhongming/930/CGMformer/output/output_zhao/97_CV1_T12D_F4_1_240106_231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L97_B48_LR0.0004_LScosine_WU20_E20_Oadamw_F4/checkpoint-400'

# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/CV_2/downsampled_CV_2_train_96"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/CV_2/downsampled_CV_2_test_96"

# train_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_finetune"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_test"

# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230805_193032_Shanghai_train_1581_288_m3060_sz1_L8_emb256_SL288_E2000_B12_LR8e-06_LSlinear_WU4000_Oadamw_DS2/checkpoint-66000" # 2000epoch sz1
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230803_112935_Shanghai_train_1981_m45_rank_L8_emb256_SL288_E300_B12_LR6e-06_LSlinear_WU500_Oadamw_DS2/checkpoint-9000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230803_014715_Shanghai_train_1981_Fre_45_80_sz10_gluVal_L8_emb256_SL288_E200_B12_LR6e-06_LSlinear_WU400_Oadamw_DS2/checkpoint-5000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230803_171657_Shanghai_train_1981_m3060_L8_emb256_SL288_E500_B8_LR6e-06_LSlinear_WU500_Oadamw_DS2/checkpoint-25000" # 配200
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230804_025709_Shanghai_train_1581_m3060_L8_emb256_SL288_E1000_B12_LR6e-06_LSlinear_WU2000_Oadamw_DS2/checkpoint-32000" # 新划分288

# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230804_123245_Shanghai_train_old288_m3060_L8_emb256_SL288_E1000_B12_LR6e-06_LSlinear_WU2000_Oadamw_DS2/checkpoint-29000" # old288
# train_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_finetune"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/Shanghai_test"

# # 新划分,downsampling96
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230804_131837_Shanghai_train_1581_96_m3060_L8_emb256_SL96_E1000_B12_LR6e-06_LSlinear_WU2000_Oadamw_DS2/checkpoint-99000"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_finetune_96"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_test_96"

# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230730_225550_Fre_45_80_sz10_L8_emb256_SL96_E200_B16_LR6e-06_LSlinear_WU1000_Oadamw_DS2/checkpoint-7000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230729_180434_TFIDF_15_60_L8_emb256_SL96_E200_B8_LR6e-06_LSlinear_WU3000_Oadamw_DS2/checkpoint-20000"
# output_path = '/share/home/liangzhongming/930/CGMformer/downStreamOutput/data_volum'
output_path = '/share/home/liangzhongming/930/CGMformer/output/output_zhao'

# load train dataset
trainset = load_from_disk(train_path)
# load evaluation dataset
testset = load_from_disk(test_path)

trainset = trainset.shuffle(seed_num)
testset = testset.shuffle(seed_num)

# rename columns
trainset = trainset.rename_column("microvascular", "label")
testset = testset.rename_column("microvascular", "label")

# create dictionary of cell types : label ids
target_names = set(list(Counter(trainset["label"]).keys()) + list(Counter(testset["label"]).keys()))
# target_names = set(list(Counter(trainset["label"]).keys())) # UNK只在测试集
target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))

# target_test_names = set(list(Counter(trainset["label"]).keys())) # UNK只在测试集
# target_test_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))

# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example
labeled_trainset = trainset.map(classes_to_ids, num_proc=16)
labeled_testset = testset.map(classes_to_ids, num_proc=16)

# filter dataset for labels in corresponding training set
trained_labels = list(Counter(labeled_trainset["label"]).keys())
def if_trained_label(example):
    return example["label"] in trained_labels
labeled_testset = labeled_testset.filter(if_trained_label, num_proc=16)

# set model parameters
# max input size
max_input_size = 289

# # set training parameters
# # max learning rate
# max_lr = 4.1e-4 # 4.1e-4 4e-6 # 5e-5
# # how many pretrained layers to freeze
# freeze_layers = 0 # 0
# # number gpus
# num_gpus = 1
# # number cpu cores
# num_proc = 16
# # batch size for training and 
# batch_size = 12
# # learning schedule
# lr_schedule_fn = "cosine"
# # warmup steps
# warmup_steps = 1385 # 100 1385
# # number of epochs
# epochs = 20
# optimizer = "adamw"



# # Zhao multi-label
# max_lr = 4.0e-5 # 4.1e-4 4e-6 # 5e-5
# # how many pretrained layers to freeze
# freeze_layers = 0 # 0
# # number gpus
# num_gpus = 1
# # number cpu cores
# num_proc = 16
# # batch size for training and 
# batch_size = 12
# # learning schedule
# lr_schedule_fn = "cosine"
# # warmup steps
# warmup_steps = 100 # 100 1385
# # number of epochs
# epochs = 20
# optimizer = "adamw"


# Zhao T1D/T2D complication
max_lr = 4e-4 # 4.1e-4 4e-6 # 5e-5
# how many pretrained layers to freeze
freeze_layers = 0 # 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and 
batch_size = 48
# learning schedule
lr_schedule_fn = "cosine"
# warmup steps
warmup_steps = 100 # 100 1385
# number of epochs
epochs = 20
optimizer = "adamw"


# # Zhao T1D/T2D linear
# max_lr = 4.0e-4 # 4.1e-4 4e-6 # 5e-5
# # how many pretrained layers to freeze
# freeze_layers = 4 # 0
# # number gpus
# num_gpus = 1
# # number cpu cores
# num_proc = 16
# # batch size for training and 
# batch_size = 48
# # learning schedule
# lr_schedule_fn = "cosine"
# # warmup steps
# warmup_steps = 20 # 100 1385
# # number of epochs
# epochs = 30
# optimizer = "adamw"

# # Zhao T1D/T2D full-finetune
# max_lr = 4.0e-4 # 4.1e-4 4e-6 # 5e-5
# # how many pretrained layers to freeze
# freeze_layers = 0 # 0
# # number gpus
# num_gpus = 1
# # number cpu cores
# num_proc = 16
# # batch size for training and 
# batch_size = 96
# # learning schedule
# lr_schedule_fn = "cosine"
# # warmup steps
# warmup_steps = 20 # 100 1385
# # number of epochs
# epochs = 20
# optimizer = "adamw"

subtask_trainset = labeled_trainset
subtask_testset = labeled_testset
subtask_label_dict = target_name_id_dict
# set logging steps

# reload pretrained model
model = BertForSequenceClassification.from_pretrained(
    checkpoint_1917_path,
    num_labels=len(subtask_label_dict.keys()),
    output_attentions=False,
    output_hidden_states=False, # False->True
).to("cuda")

if freeze_layers is not None:
    modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

# define output directory path
decs = "289_CV1_mic_PAD_F0"
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
output_dir = output_path + f"/{decs}_{datestamp}_{checkpoint_l97_path.split('/')[-2]}_L{max_input_size}_B{batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

logging_steps = round(len(labeled_trainset)/batch_size/10)
# set training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    # # "evaluation_strategy": "epoch",
    "evaluation_strategy": "steps",
    "eval_steps": logging_steps,
    # "save_strategy": "epoch",
    "save_strategy": "steps",
    "save_steps": logging_steps,
    "logging_steps": 10,
    # "group_by_length": True,
    # "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    # "lr_scheduler_power": 0.65,
    "warmup_steps": warmup_steps,
    # "dropout_rate": 0.11,
    # "gradient_clipping": 1.61,
    "weight_decay": 0.10, # 0.10
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": output_dir,
    "include_inputs_for_metrics": True
}

training_args_init = TrainingArguments(**training_args)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # preds = pred.predictions[0].argmax(-1) # for output_hidden_states=True
    input = pred.inputs

    wrong_indices = preds != labels
    wrong_samples_batch = input[wrong_indices].tolist() # convert to list for JSON
   
    wrong_preds_batch = preds[wrong_indices].tolist()
    wrong_labels_batch = labels[wrong_indices].tolist()
    wrong = {
        "samples": wrong_samples_batch,
        "preds": wrong_preds_batch,
        "labels": wrong_labels_batch
    }

    # calculate accuracy and macro f1 using sklearn's function
    # all_labels = [0, 1, 2]
    # conf_mat = confusion_matrix(labels, preds, labels=all_labels)
    conf_mat = confusion_matrix(labels, preds)

    # non_unk_indices = labels != 0
    # acc = accuracy_score(labels[non_unk_indices], preds[non_unk_indices])
    # macro_f1 = f1_score(labels[non_unk_indices], preds[non_unk_indices], average='macro')
    
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')

    # class_names = ['0', '1', '2']
    class_names = ['0', '1']
    # class_names = ['0', '1', '2']
    classwise_scores = {}
    misclassified = {}
    for i, label in enumerate(class_names):
        precision = conf_mat[i,i] / conf_mat[:,i].sum()
        recall = conf_mat[i,i] / conf_mat[i,:].sum() 
        classwise_scores[label] = {
            'precision': precision,
            'recall': recall
        }
        # if conf_mat[:,i].sum() != 0:
        #     precision = conf_mat[i,i] / conf_mat[:,i].sum()
        # else:
        #     precision = float('nan') # or some other value that represents undefined

        # if conf_mat[i,:].sum() != 0:
        #     recall = conf_mat[i,i] / conf_mat[i,:].sum()
        # else:
        #     recall = float('nan') # or some other value that represents undefined

        classwise_scores[label] = {
            'precision': precision,
            'recall': recall
        }
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'confusion_matrix': conf_mat.tolist(),
        'classwise_scores': classwise_scores,
        # 'misclassified': misclassified,
        "wrong": wrong
    }
# create the trainer
trainer = Trainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=subtask_trainset,
    eval_dataset=subtask_testset,
    compute_metrics=compute_metrics
)
# trainer = ClasssifyTrainer(
#     model=model,
#     args=training_args_init,
#     data_collator=DataCollatorForCellClassification(),
#     train_dataset=subtask_trainset,
#     eval_dataset=subtask_testset,
#     compute_metrics=compute_metrics
# )
# train the label classifier
trainer.train()

# Full Sample Test
# trainer = Trainer(
#     model=model,
#     args=training_args_init,
#     data_collator=DataCollatorForCellClassification(),
#     train_dataset=subtask_trainset,
#     eval_dataset=subtask_testset,
#     compute_metrics=compute_metrics
# )

# test
print(f"start predict!!!")
predictions = trainer.predict(subtask_testset)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)
