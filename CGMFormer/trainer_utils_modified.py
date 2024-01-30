'''
@Author: liangzhongming
@Date: 
LastEditors: Please set LastEditors
LastEditTime: 2023-09-17 00:38:59
@Description: 请填写简介
'''
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datasets import Dataset


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        # wrong_samples: Optional[np.ndarray] = None,
        # true_labels: Optional[np.ndarray] = None,
        # predicted_labels: Optional[np.ndarray] = None
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        # self.wrong_samples = wrong_samples
        # self.true_labels = true_labels
        # self.predicted_labels = predicted_labels
        

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs
        # elif idx == 3:
        #     return self.wrong_samples
        # elif idx == 4:
        #     return self.true_labels
        # elif idx == 5:
        #     return self.predicted_labels


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    wrong_samples_info: Optional[List[Dict[str, Any]]] = None # by lzm


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    dataset_df: Optional[pd.DataFrame]
