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
import math
import subprocess
import seaborn as sns
import numpy as np
import sys
import debugpy

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import scale
from scipy.stats import pearsonr
from scipy import spatial
sns.set()
from datasets import load_from_disk
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer
from transformers.training_args import TrainingArguments

from CGMFormer import BertForSequenceClassification, BertForRegression
from CGMFormer import DataCollatorForCellClassification
from CGMFormer import DataCollatorForRegressiong
from CGMFormer import ClasssifyTrainer

# debugpy.listen(("192.168.72.58", 5681))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

seed_num = 59
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 59
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_8_data/288/Shanghai_total_token2id.pkl'
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dictionary = pickle.load(f)

# 97
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir813/models/230815_182104_TFIDF4560_SZ1_clsV2_L4_H8_emb128_SL97_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# 80w 288
checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# 50w
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230829_145239_TFIDF4560_sincos_SZ1_L4_H4_emb128_SL289_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_5/train"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_5/test"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/96/downsampled_CV_1_train_96"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/96/downsampled_CV_1_test_96"

# CITY
train_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_origin_train"
test_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_origin_test"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_Interpolated_train"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_Interpolated_test"

# # Colas
# train_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas_train"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas_test"

output_path = '/share/home/liangzhongming/930/CGMformer/output_reg'
# load train dataset
trainset = load_from_disk(train_path)
# load evaluation dataset
testset = load_from_disk(test_path)

trainset = trainset.shuffle(seed_num)
testset = testset.shuffle(seed_num)

target_label = "hba1c"
# homa-b:0-400
# homa-is:0.2-5
trainset = trainset.filter(
    lambda example: 
        not (example[target_label] is None or 
             math.isnan(example[target_label]))
            #  example[target_label] < 0.2 or
            #  example[target_label] > 5)
)
testset = testset.filter(
    lambda example: 
        not (example[target_label] is None or 
             math.isnan(example[target_label]))
            #  example[target_label] < 0.2 or
            #  example[target_label] > 5)
)

train_labels = trainset[target_label]
# scaled_train_labels = scale(train_labels) 
scaled_train_labels = train_labels
scaled_trainset = trainset.add_column("scaled_label", scaled_train_labels)

test_labels = testset[target_label]
# scaled_test_labels = scale(test_labels, with_mean=True, with_std=True) # Using the mean and standard deviation of the training set to scale the test set
scaled_test_labels = test_labels

scaled_testset = testset.add_column("scaled_label", scaled_test_labels)

# rename columns
labeled_trainset = scaled_trainset.rename_column("scaled_label", "label")
labeled_testset = scaled_testset.rename_column("scaled_label", "label")
# # rename columns
# labeled_trainset = trainset.rename_column(target_label, "label")
# labeled_testset = testset.rename_column(target_label, "label")

# set model parameters
# max input size
max_input_size = 289

# set training parameters
# max learning rate
max_lr = 5.5e-6 # 4e-6 # 5e-5
# how many pretrained layers to freeze
freeze_layers = 0 # 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and 
batch_size = 12
# learning schedule
lr_schedule_fn = "cosine"
# warmup steps
warmup_steps = 1392
# weight_decay
weight_decay = 0.11
# number of epochs
epochs = 50
optimizer = "adamw"

subtask_trainset = labeled_trainset
subtask_testset = labeled_testset
# set logging steps

# reload pretrained model
# model = BertForSequenceClassification.from_pretrained(
#     checkpoint_path,
#     num_labels=1, # for regression
#     output_attentions=False,
#     output_hidden_states=False,
# ).to("cuda")
model = BertForRegression.from_pretrained(
    checkpoint_path,
    num_labels=1, # for regression
    output_attentions=False,
    output_hidden_states=False,
).to("cuda")


if freeze_layers is not None:
    modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False


# define output directory path
decs = "CITY_origin_hba1c_80w_913_F0_Unscale"
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
output_dir = output_path + f"/{decs}_{datestamp}_{checkpoint_path.split('/')[-2]}_L{max_input_size}_B{batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"

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
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay, # 0.001->0.01
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
    preds = pred.predictions
    input = pred.inputs

    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    # preds_nonzero = preds[labels!=0]
    preds_reval = preds[:,0].ravel()
    preds_nonzero = preds_reval[labels!=0]
    labels_nonzero = labels[labels!=0]  

    # mape = mean_absolute_percentage_error(labels_nonzero, preds_nonzero)

    pearson_corr, _ = pearsonr(labels, preds_reval)

    return {
        'mse': mse,
        'r2': r2,
        'pearson_corr': pearson_corr
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
#     data_collator=DataCollatorForRegressiong(),
#     train_dataset=subtask_trainset,
#     eval_dataset=subtask_testset,
#     compute_metrics=compute_metrics
# )
# train the label classifier
trainer.train()

# test
print(f"start predict!!!")
predictions = trainer.predict(subtask_testset)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)
