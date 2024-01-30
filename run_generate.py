'''
@Author: liangzhongming
@Date: 
LastEditors: Please set LastEditors
LastEditTime: 2024-01-26 14:02:31
@Description: 请填写简介
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

from CGMFormer import BertForSequenceClassification
from CGMFormer import BertWithGeneration
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

# 80w parameters
checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# 50w parameters
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230829_145239_TFIDF4560_sincos_SZ1_L4_H4_emb128_SL289_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"

# checkpoint
classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-1248"
total_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_total"

train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_1/train"
test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_1/test"

output_path = '/share/home/liangzhongming/930/CGMformer/downStreamOutput/819'
# load train dataset
trainset = load_from_disk(train_path)
# load evaluation dataset
testset = load_from_disk(test_path)

trainset = trainset.shuffle(seed_num)
testset = testset.shuffle(seed_num)

# def format_labels(example):
#     labels = [example[label] for label in ['id', 'types', 'age', 'bmi', 'hba1c', 'homa-b', 'homa-is', 'index', 'Fast_s', 'Fast_e', 'Dawn_s', 'Dawn_e', 'Breakfast_s', 'Breakfast_e', 'Lunch_s', 'Lunch_e', 'Dinner_s', 'Dinner_e']]
#     example['label'] = labels
#     return example

# labeled_trainset = trainset.map(format_labels, num_proc=16)
# labeled_testset = testset.map(format_labels, num_proc=16)

# 
label_columns = ['id', 'types', 'age', 'bmi', 'hba1c', 'homa-b', 'homa-is', 'index', 'Fast_s', 'Fast_e', 'Dawn_s', 'Dawn_e', 'Breakfast_s', 'Breakfast_e', 'Lunch_s', 'Lunch_e', 'Dinner_s', 'Dinner_e']
labeled_trainset = trainset.select(label_columns)
labeled_testset = testset.select(label_columns)

# set model parameters
# max input size
max_input_size = 289

# set training parameters
# max learning rate
max_lr = 4.1e-4 # 4.1e-4 4e-6 # 5e-5
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
warmup_steps = 1385 # 100 1385
# number of epochs
epochs = 20
optimizer = "adamw"

subtask_trainset = labeled_trainset
subtask_testset = labeled_testset
# set logging steps

# reload pretrained model
model = BertWithGeneration.from_pretrained(
    checkpoint_path,
    output_attentions=False,
    output_hidden_states=False, # False->True
).to("cuda")


if freeze_layers is not None:
    modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

 

# define output directory path
decs = "generate_seq"
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

    # # 含UNK标签的样本
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

# test
print(f"start predict!!!")
predictions = trainer.predict(subtask_testset)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)
