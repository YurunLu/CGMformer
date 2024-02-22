# CGMformer

CGMformer : A generative pretrained transformer for predicting and decoding individual glucose dynamics from continuous glucose monitoring data 


## Installation

```
pip install -r requirements.txt
```



## Data processing

Please refer to `processing_82_data.ipynb`、`processing_89_data.ipynb`、`processing_811_data.ipynb`、`build_vocab.ipynb` to process the CGM data into a format accepted by the model

## Pre-training
### Pre-training CGMformer

To train CGMformer using unlabeled CGM data, use the `run_pretrain_CGMFormer.py` script.

```
deepspeed --num_gpus={num_gpus} run_pretrain_CGMFormer.py
```

where
- `num_gpus`: number of GPUs used for training

### Getting sample embeddings without fine-tuning

```
python run_clustering.py --checkpoint_path /path/to/checkpoint --data_path /path/to/data --save_path /path/to/save
```

## Diagnosis 

```
python run_labels_classify.py --checkpoint_path /path/to/checkpoint --train_path /path/to/train_data --test_path /path/to/test_data --output_path /path/to/save
```

## CGMformer_C
To training CGMformer_C, paired CGM data and clinical data including `age, bmi, fpg, ins0, HOMA-IS, HOMA-B, pg120, hba1c, hdl` are needed:
```
python SupervisedC.py
```
To calculate CGMformer_C from trained model and embedded vectors from CGMformer:
```
python CalculateSC.py
```
## CGMformer_type
CGMformer_type provides subtyping based on CGM data. Embedded vectors from CGMformer are required.
```
python Classifier.py
```

## CGMformer_Diet
Paired embedded vector, meal nutrition information, and before (and post) meal glucose are required for (training) CGMformer_Diet:
```
python PredictGlucose.py
```
