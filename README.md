# CGMformer

CGMformer : A pretrained transformer model for decoding individual glucose dynamics from continuous glucose monitoring data.

The article is available at:
Yurun Lu, Dan Liu, Zhongming Liang, et al. A pretrained transformer model for decoding individual glucose dynamics from continuous glucose monitoring data. National Science Review https://doi.org/10.1093/nsr/nwaf039

## Installation

```
pip install -r requirements.txt
```



## Data processing

Different CGM data have different attributes, we recommend to refer to the `processing_811_data.ipynb` to process your data, where the continuous glucose data are labeled with the key "input_ids".

In `build_vocab.ipynb`, we generate a vocab from 39-301 and containing `<MASK>`, `<PAD>`, `<CLS>` token.

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
