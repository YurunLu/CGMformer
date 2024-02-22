# CGMformer

CGMformer : A generative pretrained transformer for predicting and decoding individual glucose dynamics from continuous glucose monitoring data 


## Installation

```
pip install -r requirements.txt
```



## Data processing

Please refer to `processing_82_data.ipynb`、`processing_89_data.ipynb`、`processing_811_data.ipynb`、`build_vocab.ipynb` to process the CGM data into a format accepted by the model



## Pre-training CGMformer

To train CGMformer using unlabeled CGM data, use the `run_pretrain_CGMFormer.py` script.

```
deepspeed --num_gpus={num_gpus} run_pretrain_CGMFormer.py
```

where
- `num_gpus`: number of GPUs used for training



## Getting sample embeddings without fine-tuning

```
python run_clustering.py
```



## Multi-label

```
python run_mutil_labels.py
```



## Multi-class
```
python run_labels_classify.py
```



## Regression
```
python run_regression.py
```



## Support

If you have any questions, please feel free to contact us  



## Citation
