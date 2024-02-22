
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
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint')
    parser.add_argument('--train_path', type=str, help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, help='Path to the testing dataset')
    parser.add_argument('--output_path', type=str, help='Path to save output files')
    return parser.parse_args()
    
def main():
    seed_num = 51
    random.seed(seed_num)
    np.random.seed(seed_num)
    seed_val = 51
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl' # Repalce it with your token_dictionary
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dictionary = pickle.load(f)
    
    # output paths
    output_path = args.output_path
    # load train dataset
    trainset = load_from_disk(args.train_path)
    # load evaluation dataset
    testset = load_from_disk(args.test_path)
    
    trainset = trainset.shuffle(seed_num)
    testset = testset.shuffle(seed_num)
    
    # rename columns
    trainset = trainset.rename_column("microvascular", "label")
    testset = testset.rename_column("microvascular", "label")
    
    # create dictionary of cell types : label ids
    target_names = set(list(Counter(trainset["label"]).keys()) + list(Counter(testset["label"]).keys()))
    # target_names = set(list(Counter(trainset["label"]).keys())) 
    target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
    
    # target_test_names = set(list(Counter(trainset["label"]).keys())) 
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
    max_input_size = 289 # 288 + 'CLS'
    
    # # set training parameters
    # # max learning rate
    # max_lr = 4.1e-4 
    # # how many pretrained layers to freeze
    # freeze_layers = 0 
    # # number gpus
    # num_gpus = 1
    # # number cpu cores
    # num_proc = 16
    # # batch size for training and 
    # batch_size = 12
    # # learning schedule
    # lr_schedule_fn = "cosine"
    # # warmup steps
    # warmup_steps = 1385 
    # # number of epochs
    # epochs = 20
    # optimizer = "adamw"
    
    
    
    # # Zhao multi-label
    # max_lr = 4.0e-5 
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
    max_lr = 4e-4 
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
    # max_lr = 4.0e-4 
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
    # max_lr = 4.0e-4 
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
        args.checkpoint_path,
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
        # "evaluation_strategy": "epoch",
        "evaluation_strategy": "steps",
        "eval_steps": logging_steps,
        # "save_strategy": "epoch",
        "save_strategy": "steps",
        "save_steps": logging_steps,
        "logging_steps": 10,
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
    
if __name__ == "__main__":
    main()
