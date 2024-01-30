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
import pickle
import torch
import random
import subprocess
import seaborn as sns
import numpy as np
import sys
import debugpy
from scipy.special import expit

sns.set()
from datasets import load_from_disk
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
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

# complications
checkpoint_path = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
train_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_1/train"
test_path = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/Zhao/CV_1/test"

output_path = '/share/home/liangzhongming/930/CGMformer/downStreamOutput/819'
# load train dataset
trainset = load_from_disk(train_path)
# load evaluation dataset
testset = load_from_disk(test_path)

trainset = trainset.shuffle(seed_num)
testset = testset.shuffle(seed_num)

def format_labels(example):
    labels = [example[label] for label in ['cerebrovascular_disease', 'coronary_heart_disease', 'peripheral_arterial_disease', 'nephropathy', 'neuropathy', 'retinopathy', 'macrovascular ', 'microvascular', 'complication']]
    example['label'] = labels
    return example

labeled_trainset = trainset.map(format_labels)
labeled_testset = testset.map(format_labels)


# how many pretrained layers to freeze
freeze_layers = 0 # 0
# batch size for training and eval
batch_size = 24
# number of epochs
epochs = 10

# logging steps
logging_steps = round(len(labeled_trainset)/batch_size/10)

# define function to initiate model
def model_init():
    model = BertForSequenceClassification.from_pretrained(checkpoint_path,
                                                          num_labels=0,
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
decs = "hyperopt_search_mutillabels_828"
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
    # preds = pred.predictions.argmax(-1)
    preds_array = pred.predictions
    # probs = expit(preds_array)
    probs = 1 / (1 + np.exp(-preds_array))  # Apply the Sigmoid function
    preds = (probs > 0.5).astype(int)
    input = pred.inputs

    # Compute metrics for multi-label classification
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    macro_precision = precision_score(labels, preds, average='macro')
    macro_recall = recall_score(labels, preds, average='macro')
    

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
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
# trainer = Trainer(
#     model_init=model_init,
#     args=training_args_init,
#     data_collator=DataCollatorForCellClassification(),
#     train_dataset=labeled_trainset,
#     eval_dataset=labeled_testset,
#     compute_metrics=compute_metrics
# )
# trainer = ClasssifyTrainer(
#     model_init=model_init,
#     args=training_args_init,
#     data_collator=DataCollatorForCellClassification(),
#     train_dataset=labeled_trainset,
#     eval_dataset=labeled_testset,
#     compute_metrics=compute_metrics
# )
trainer = Trainer(
    model_init=model_init,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=labeled_trainset,
    eval_dataset=labeled_testset,
    compute_metrics=compute_metrics
)


# specify raytune hyperparameter search space
ray_config = {
    "num_train_epochs": tune.choice([20, 50, 100]),
    "learning_rate": tune.loguniform(1e-6, 1e-3),
    "weight_decay": tune.uniform(0.0, 0.3),
    "lr_scheduler_type": tune.choice(["linear","cosine","polynomial"]),
    "warmup_steps": tune.uniform(10, 2000),
    "seed": tune.uniform(0,100),
    "per_device_train_batch_size": tune.choice([12, 24]),
    # "dropout_rate": tune.uniform(0.0, 0.5), 
    # "gradient_clipping": tune.uniform(0, 2), 
    # "adam_beta1": tune.uniform(0.8, 0.99), 
    # "adam_beta2": tune.uniform(0.8, 0.999), 
    # "adam_epsilon": tune.loguniform(1e-9, 1e-6), 
}

hyperopt_search = HyperOptSearch(
    metric="eval_accuracy", mode="max")

# optimize hyperparameters
trainer.hyperparameter_search(
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
                                                   metric="eval_accuracy",
                                                   metric_columns=["loss", "eval_loss", "eval_accuracy"])
)

# analysis = ExperimentAnalysis(trainer.get_experiment().results)


# best_trial = analysis.get_best_trial("eval_accuracy", mode="max", scope="all") 

# best_config = best_trial.config

# print("Best hyperparameters:", best_config)
