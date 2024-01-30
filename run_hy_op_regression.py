import os

GPU_NUMBER = [0, 1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

# imports
# initiate runtime environment for raytune
import pyarrow # must occur prior to ray import
import ray
from ray import tune
from ray.tune import ExperimentAnalysis
# from ray.tune.suggest import HyperOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

from collections import Counter
import datetime
import math
import json
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import scale
from scipy.stats import pearsonr
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer
from transformers.training_args import TrainingArguments

from CGMFormer import BertForSequenceClassification
from CGMFormer import DataCollatorForCellClassification
from CGMFormer import ClasssifyTrainer

# debugpy.listen(("192.168.72.58", 5681))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

num_proc=32

seed_num = 59
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 59
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dictionary = pickle.load(f)

checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
# train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_4/train"
# test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_4/test"

# CITY
train_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_origin_train"
test_path = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/CITY_origin_test"

output_path = '/share/home/liangzhongming/930/CGMformer/downStreamOutput/828'
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
scaled_train_labels = scale(train_labels) 
scaled_trainset = trainset.add_column("scaled_label", scaled_train_labels)
 
test_labels = testset[target_label]
scaled_test_labels = scale(test_labels, with_mean=True, with_std=True) # 使用训练集的均值和标准差来scale
scaled_testset = testset.add_column("scaled_label", scaled_test_labels)

# rename columns
labeled_trainset = scaled_trainset.rename_column("scaled_label", "label")
labeled_testset = scaled_testset.rename_column("scaled_label", "label")


# how many pretrained layers to freeze
freeze_layers = 0 # 0
# batch size for training and eval
batch_size = 24
# number of epochs
epochs = 1

subtask_trainset = labeled_trainset
subtask_testset = labeled_testset
# set logging steps
# logging steps
logging_steps = round(len(labeled_trainset)/batch_size/10)

# define function to initiate model
def model_init():
    model = BertForSequenceClassification.from_pretrained(checkpoint_path,
                                                          num_labels=1,
                                                          output_attentions = False,
                                                          output_hidden_states = False)
    if freeze_layers is not None:
        modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    model = model.to("cuda")
    return model


# define output directory path
decs = "hyperopt_search_CITY_hba1c"
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
output_dir = output_path + f"/{decs}_{datestamp}_{checkpoint_path.split('/')[-2]}/"

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# set training arguments
training_args = {
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
    "disable_tqdm": True,
    "skip_memory_metrics": True, # memory tracker causes errors in raytune
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

def compute_metricsV1(pred):
    labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    preds = pred.predictions[0].argmax(-1) # for output_hidden_states=True
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
    all_labels = [0, 1, 2]
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
    model_init=model_init,
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

# specify raytune hyperparameter search space
ray_config = {
    "num_train_epochs": tune.choice([20, 50]),
    "learning_rate": tune.loguniform(1e-8, 1e-3),
    "weight_decay": tune.uniform(0.0, 0.3),
    "lr_scheduler_type": tune.choice(["linear","cosine","polynomial"]),
    "warmup_steps": tune.uniform(100, 2000),
    "seed": tune.uniform(0, 100),
    "per_device_train_batch_size": tune.choice([12, 24]),
    # "dropout_rate": tune.uniform(0.0, 0.5), 
    # "gradient_clipping": tune.uniform(0, 2), 
    # "adam_beta1": tune.uniform(0.8, 0.99), 
    # "adam_beta2": tune.uniform(0.8, 0.999), 
    # "adam_epsilon": tune.loguniform(1e-9, 1e-6), 
}

hyperopt_search = HyperOptSearch(
    metric="eval_pearson_corr", mode="max")

# optimize hyperparameters
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    resources_per_trial={"cpu":16,"gpu":1},
    hp_space=lambda _: ray_config,
    search_alg=hyperopt_search,
    n_trials=100, # number of trials
    progress_reporter=tune.CLIReporter(max_report_frequency=600,
                                                   sort_by_metric=True,
                                                   max_progress_rows=100,
                                                   mode="max",
                                                   metric="eval_pearson_corr",
                                                   metric_columns=["loss", "eval_loss", "eval_r2", "eval_mse", "eval_pearson_corr"])
)
print(best_run.get_best_config(metric="eval_pearson_corr", mode="max"))
best_hyperparameters = best_run.hyperparameters
with open("/share/home/liangzhongming/930/CGMformer/output_reg/homa-is_best_run.json", "w") as f:
    json.dump(best_hyperparameters, f)
