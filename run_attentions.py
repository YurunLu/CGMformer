'''
Author: lzm 1015511952@qq.com
Date: 2023-10-08 15:53:52
LastEditors: Please set LastEditors
LastEditTime: 2024-01-26 13:59:47
FilePath: /CGMformer/run_attentions.py
Description: 
'''
import os

GPU_NUMBER = [0, 1]
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
from bertviz import head_view, model_view

from CGMFormer import BertForSequenceClassification
from CGMFormer import DataCollatorForCellClassification
from CGMFormer import ClasssifyTrainer

debugpy.listen(("192.168.72.59", 5681))
print("Waiting for debugger attach...")
debugpy.wait_for_client()

seed_num = 51
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 51
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dictionary = pickle.load(f)


# classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-1248"
# 0.84
# classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-520"
# 0.86
classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-546"

# classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-533"

# total_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_total"
visualization_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_1/test"
# total_dataset = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas"

# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230730_225550_Fre_45_80_sz10_L8_emb256_SL96_E200_B16_LR6e-06_LSlinear_WU1000_Oadamw_DS2/checkpoint-7000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230729_180434_TFIDF_15_60_L8_emb256_SL96_E200_B8_LR6e-06_LSlinear_WU3000_Oadamw_DS2/checkpoint-20000"
output_path = '/share/home/liangzhongming/930/CGMformer/downStreamOutput/819'

totalData = load_from_disk(visualization_dataset)

# rename columns
totalData = totalData.rename_column("types", "label")

# # create dictionary of cell types : label ids
# target_names = set(list(Counter(totalData["label"]).keys()))
# # target_names = set(list(Counter(trainset["label"]).keys()))
# target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))

target_names = {'IGR', 'NGT', 'T2D'}
target_name_id_dict = {'IGR': 0, 'NGT': 1, 'T2D': 2}

# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example
labeled_totalData = totalData.map(classes_to_ids, num_proc=16)

# set model parameters
# max input size
max_input_size = 289

# set training parameters
# max learning rate
max_lr = 4.1e-4 # 4.1e-4 4e-6 # 5e-5
# how many pretrained layers to freeze
freeze_layers = 4 # 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and 
batch_size = 8
# learning schedule
lr_schedule_fn = "cosine"
# warmup steps
warmup_steps = 1385 # 100 1385
# number of epochs
epochs = 20
optimizer = "adamw"

subtask_totalDataset = labeled_totalData

subtask_label_dict = target_name_id_dict
# set logging steps

# reload pretrained model
model = BertForSequenceClassification.from_pretrained(
    classify_checkpoint_path,
    num_labels=len(subtask_label_dict.keys()), # Colas need + 1
    output_attentions=True,
    output_hidden_states=False, # False->True
    # ignore_mismatched_sizes=True,
).to("cuda")

if freeze_layers is not None:
    modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

# outputs = model(**inputs, output_attentions=True)

# define output directory path
decs = "attentions"
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
output_dir = output_path + f"/{decs}_{datestamp}_{classify_checkpoint_path.split('/')[-2]}_L{max_input_size}_B{batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

logging_steps = round(len(subtask_totalDataset)/batch_size/10)

# set training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": False,
    "do_eval": False,
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
    # preds = pred.predictions.argmax(-1)
    preds = pred.predictions[0].argmax(-1) # for output_hidden_states=True or output_attentions=True
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
    # class_names = ['0', '1']
    class_names = ['0', '1', '2']
    classwise_scores = {}
    misclassified = {}
    for i, label in enumerate(class_names):
        precision = conf_mat[i,i] / conf_mat[:,i].sum()
        recall = conf_mat[i,i] / conf_mat[i,:].sum() 
        classwise_scores[label] = {
            'precision': precision,
            'recall': recall
        }

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
# trainer = Trainer(
#     model=model,
#     args=training_args_init,
#     data_collator=DataCollatorForCellClassification(),
#     compute_metrics=compute_metrics
# )
trainer = ClasssifyTrainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    compute_metrics=compute_metrics
)

# test
print(f"start predict!!!")
predictions = trainer.predict(subtask_totalDataset)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)
