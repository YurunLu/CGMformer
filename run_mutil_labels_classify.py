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
from scipy.special import expit
import debugpy

sns.set()
from datasets import load_from_disk
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, fbeta_score, label_ranking_average_precision_score, jaccard_score, precision_recall_fscore_support, hamming_loss
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer
from transformers.training_args import TrainingArguments
from transformers.configuration_utils import PretrainedConfig

from CGMFormer import BertForSequenceClassification
from CGMFormer import DataCollatorForCellClassification
from CGMFormer import ClasssifyTrainer

# debugpy.listen(("192.168.72.59", 5682))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

seed_num = 59
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 59
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/CV_2/CV2_total_token2id.pkl'
# TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_8_data/288/Shanghai_total_token2id.pkl'
# TOKEN_DICTIONARY_FILE = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_vocab_rank.pkl"
TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dictionary = pickle.load(f)

# paths
# train_path = "/share/home/liangzhongming/930/downstream_cl_data/train"
# test_path = "/share/home/liangzhongming/930/downstream_cl_data/train/test"

# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_finetune"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_test"

# Non-UNK 5 cv
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230805_185023_Shanghai_train_1581_288_m3060_L8_emb256_SL288_E2000_B12_LR8e-06_LSlinear_WU4000_Oadamw_DS2/checkpoint-66000" # 2000epoch sz10
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230805_224807_Shanghai_train_1581_288_m3060_sz1_L8_emb256_SL288_E3000_B12_LR2e-05_LSlinear_WU5000_Oadamw_DS2/checkpoint-99000" # 新288 3000epoch sz1
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_010530_Fre3060_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # old288 3000epoch sz1
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_010530_Fre3060_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230809_003104_TFIDF4560_SZ1_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230810_002350_Fre3060_SZ1_L8_emb256_SL96_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # 96 m3060
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir88/models/230810_131518_Fre3060_SZ10_L8_emb256_SL288_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-33000" # tokenizer10 m3060

# 288
checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"

# 96
checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000" # 820

# 9labels
classify9_best_checkpoint = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/1028/mulit_9labels_F0_CV2_1109_231109_231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_mean_last1_L97_B24_LR2.5e-06_LScosine_WU200_E100_Oadamw_F0/checkpoint-712"
classify9_wrong_checkpoint = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/1028/mulit_9labels_F0_CV5_1109_231109_231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_mean_last1_L97_B24_LR2.5e-06_LScosine_WU200_E100_Oadamw_F0/checkpoint-712"

# 3labels
classify3_best_checkpoint = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/1028/mulit_3labels_F0_CV2_1109_231109_231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_mean_last1_L97_B24_LR2.5e-06_LScosine_WU200_E100_Oadamw_F0/checkpoint-260"
classify3_wrong_checkpoint = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/1028/mulit_3labels_F0_CV5_1109_231109_231029_020352_dim128_97_TFIDF_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_mean_last1_L97_B24_LR2.5e-06_LScosine_WU200_E100_Oadamw_F0/checkpoint-260"

# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230815_182104_TFIDF4560_SZ1_clsV2_L4_H8_emb128_SL97_E3000_B48_LR2e-05_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_5/train"
test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_5/test"
zhao_total_dataset = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/1017_Zhao_total"

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

# # New division，downsampling 96
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir84/models/230804_131837_Shanghai_train_1581_96_m3060_L8_emb256_SL96_E1000_B12_LR6e-06_LSlinear_WU2000_Oadamw_DS2/checkpoint-99000"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_finetune_96"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_200_test_96"

# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230730_225550_Fre_45_80_sz10_L8_emb256_SL96_E200_B16_LR6e-06_LSlinear_WU1000_Oadamw_DS2/checkpoint-7000"
# checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output_dir/models/230729_180434_TFIDF_15_60_L8_emb256_SL96_E200_B8_LR6e-06_LSlinear_WU3000_Oadamw_DS2/checkpoint-20000"

# output_path = '/share/home/liangzhongming/930/CGMformer/downStreamOutput/1028'
output_path = '/share/home/liangzhongming/930/CGMformer/output/output_zhao'

# load train dataset
# trainset = load_from_disk(train_path)
# # load evaluation dataset
# testset = load_from_disk(test_path)

# total_data for predict
total_data = load_from_disk(zhao_total_dataset)

# trainset = trainset.shuffle(seed_num)
# testset = testset.shuffle(seed_num)

def format_labels(example):
    # labels = [example[label] for label in ['cerebrovascular_disease', 'coronary_heart_disease', 'peripheral_arterial_disease', 'nephropathy', 'neuropathy', 'retinopathy', 'macrovascular ', 'microvascular', 'complication']]
    labels = [example[label] for label in ['macrovascular ', 'microvascular', 'complication']]
    example['label'] = labels
    return example

# labeled_trainset = trainset.map(format_labels)
# labeled_testset = testset.map(format_labels)
labeled_total = total_data.map(format_labels)


# set model parameters
# max input size
max_input_size = 97

# set training parameters
# max learning rate
max_lr = 2.5e-6 # 4e-6 # 5e-5
# how many pretrained layers to freeze
freeze_layers = 0 # 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and 
batch_size = 24
# learning schedule
lr_schedule_fn = "cosine"
# warmup steps
warmup_steps = 200
# number of epochs
epochs = 100
optimizer = "adamw"

# subtask_trainset = labeled_trainset
# subtask_testset = labeled_testset

config = {
    "problem_type": "multi_label_classification",
    # "num_labels": 9,
}

config = PretrainedConfig(**config)

# reload pretrained model
model = BertForSequenceClassification.from_pretrained(
    # checkpoint_path,
    classify3_wrong_checkpoint,
    problem_type="multi_label_classification",
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False, # False->True
).to("cuda")

if freeze_layers is not None:
    modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

# define output directory path
decs = "T12D"
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
output_dir = output_path + f"/{decs}_{datestamp}_{checkpoint_path.split('/')[-2]}_mean_last1_L{max_input_size}_B{batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# logging_steps = round(len(labeled_trainset)/batch_size/10)
logging_steps = round(len(labeled_total)/batch_size/10)

# set training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    # # "evaluation_strategy": "epoch",
    "evaluation_strategy": "steps",
    "eval_steps": logging_steps, # 200
    # "save_strategy": "epoch",
    "save_strategy": "steps",
    "save_steps": logging_steps,
    "logging_steps": 10,
    # "group_by_length": True,
    # "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.002,
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": output_dir,
    "include_inputs_for_metrics": True,
}

training_args_init = TrainingArguments(**training_args)

def compute_metrics_V3(pred):
    labels = pred.label_ids
    preds_array = pred.predictions
    probs = 1 / (1 + np.exp(-preds_array))

    threshold = 0.5
    binary_preds = (probs > threshold).astype(int)

    mutil_accuracy = accuracy_score(labels, binary_preds)
    mutil_precision = precision_score(labels, binary_preds, average='micro')
    mutil_recall = recall_score(labels, binary_preds, average='micro')
    mutil_f1 = f1_score(labels, binary_preds, average='micro')
    mutil_hamming = hamming_loss(labels, binary_preds)
    
    num_labels = labels.shape[1]
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    hamming_list = []
    for label_idx in range(num_labels):
        label_pred = binary_preds[:, label_idx]
        label_true = labels[:, label_idx]

        accuracy = accuracy_score(label_true, label_pred)
        precision = precision_score(label_true, label_pred)
        recall = recall_score(label_true, label_pred)
        f1 = f1_score(label_true, label_pred)
        hamming = hamming_loss(label_true, label_pred)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        hamming_list.append(hamming)

    return {
        'all_accuracy': mutil_accuracy,
        'all_precision': mutil_precision,
        'all_recall': mutil_recall,
        'all_f1': mutil_f1,
        'all_hamming_loss': mutil_hamming,
        
        'accuracy': accuracy_list,
        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list,
        'hamming_loss': hamming_list
    }

def compute_metricsV2(pred):
    labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    preds_array = pred.predictions
    # probs = expit(preds_array)  # Apply the Sigmoid function
    probs = 1 / (1 + np.exp(-preds_array))
    preds = (probs > 0.5).astype(int)
    input = pred.inputs
    
    # Compute metrics for multi-label classification
    acc = accuracy_score(labels, preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    metrics = {}
    for i, label in enumerate(label_list):
        metrics[f'{label}_precision'] = precision[i]  
        metrics[f'{label}_recall'] = recall[i]
        metrics[f'{label}_f1'] = f1[i]
    
    
    acc1 = accuracy_score(labels, preds, normalize=False)
    weighted_acc = accuracy_score(labels, preds, sample_weight=labels.sum(axis=1))
    acc2 = np.sum(np.all(labels == preds, axis=1)) / labels.shape[0]
    # f1_acc = fbeta_score(labels, preds, beta=1, average='samples')
    lrap = label_ranking_average_precision_score(labels, preds)
    jaccard = jaccard_score(labels, preds, average="samples") 
    
    
    
    macro_f1 = f1_score(labels, preds, average='macro')
    macro_precision = precision_score(labels, preds, average='macro')
    macro_recall = recall_score(labels, preds, average='macro')
    # micro_f1 = f1_score(labels, preds, average='micro')
    # micro_precision = precision_score(labels, preds, average='micro')
    # micro_recall = recall_score(labels, preds, average='micro')

    # classwise_scores = {}
    # for i in range(labels.shape[1]):
    #     precision = precision_score(labels[:, i], preds[:, i])
    #     recall = recall_score(labels[:, i], preds[:, i])
    #     classwise_scores[f'class_{i}'] = {
    #         'precision': precision,
    #         'recall': recall
    #     }

    return {
        'accuracy': acc,
        # 'accuracy1': acc1,
        # 'accuracy2': acc2,
        # 'accuracy3': weighted_acc,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        # 'lrap': lrap,
        # 'jaccard': jaccard,
        # 'classwise_scores': classwise_scores,
    }

def compute_metrics(pred):
    labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    preds_array = pred.predictions
    # probs = expit(preds_array)  # Apply the Sigmoid function
    probs = 1 / (1 + np.exp(-preds_array))
    preds = (probs > 0.5).astype(int)
    input = pred.inputs
    
    #
    multi_precision = [] 
    multi_recall = []
    multi_f1 = []

    single_precision = []
    single_recall = []  
    single_f1 = []

    n_labels = 9
    for i in range(n_labels):
        y_true = labels[:, i] # ground truth for one label
        y_pred = preds[:, i] # predictions for one label
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
        single_precision.append(precision)
        single_recall.append(recall) 
        single_f1.append(f1)
    
    y_true_multi = labels 
    y_pred_multi = preds

    precision = precision_score(y_true_multi, y_pred_multi, average='micro')  
    recall = recall_score(y_true_multi, y_pred_multi, average='micro')
    f1 = f1_score(y_true_multi, y_pred_multi, average='micro')

    multi_precision.append(precision)
    multi_recall.append(recall)
    multi_f1.append(f1)

    return {
        'single_precision': single_precision,
        'single_recall': single_recall,
        'single_f1': single_f1,
        'multi_precision': multi_precision,
        'multi_recall': multi_recall,
        'multi_f1': multi_f1,

    }

# create the trainer
# trainer = Trainer(
#     model=model,
#     args=training_args_init,
#     data_collator=DataCollatorForCellClassification(),
#     train_dataset=subtask_trainset,
#     eval_dataset=subtask_testset,
#     compute_metrics=compute_metrics
# )
trainer = ClasssifyTrainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    # train_dataset=subtask_trainset,
    # eval_dataset=subtask_testset,
    compute_metrics=compute_metrics_V3
)
# train the label classifier
# trainer.train()

# test
print(f"start predict!!!")
# predictions = trainer.predict(subtask_testset)
predictions = trainer.predict(labeled_total)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)
