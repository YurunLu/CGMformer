from collections.abc import Mapping
from typing import List, Union, Dict, Any, Tuple, Optional
from transformers.data.data_collator import _torch_collate_batch
import torch
import numpy as np
import pickle
from transformers import DataCollatorForLanguageModeling


TOKEN2ID_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl' # id2token 201: '<mask>'
ID2TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/id2token.pkl'
with open(TOKEN2ID_DICTIONARY_FILE, "rb") as f:
    token2id = pickle.load(f)
with open(ID2TOKEN_DICTIONARY_FILE, "rb") as f:
    id2token = pickle.load(f)

class DataCollatorForLanguageModelingModified(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):

            # for example in examples:
            #     # example['input_ids'] = [token2id[(int(token) - 30) // 10] for token in example['input_ids']] # for glu value
            #     example['input_ids'] = [token2id[int(token)] for token in example['input_ids']]
            #     # example['input_ids'] = [self.tokenizer.convert_tokens_to_ids(str(int(float(token)))) for token in example['input_ids']]
            cls_token_id = token2id['<cls>'] # 265
            pad_token_id = token2id['<pad>'] # 264
            for example in examples:
                # tokens = np.array(example['input_ids'], dtype=int)
                tokens = np.array(example['input_ids'], dtype=float) # int->float 保留nan
                nan_index = np.isnan(tokens)
                
                tokens[~nan_index] = np.clip(tokens[~nan_index], 39, 301) 
                tokens[~nan_index] = [token2id[int(token)] for token in tokens[~nan_index]]
                
                tokens[nan_index] = pad_token_id 
                tokens_with_cls = np.insert(tokens, 0, cls_token_id)

                tokens_with_cls = tokens_with_cls.astype(int)
                
                # example['input_ids'] = [token2id[token] for token in tokens_with_cls]
                example['input_ids'] = tokens_with_cls.tolist()

            input_ids = torch.tensor([example['input_ids'] for example in examples], dtype=torch.long)
            
            
            # batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            # input_ids = torch.tensor([example['input_ids'] for example in examples], dtype=torch.long)
            # values = torch.tensor([example['values'] for example in examples])
            # species= torch.tensor([example['species'] for example in examples])
            
            pad_indexes = np.where(input_ids==pad_token_id)
            attention_masks = torch.ones_like(input_ids) # 标记pad,1关注
            attention_masks[pad_indexes] = 0

            # original_lengths = (input_ids != 0).sum(dim=1)
            # attention_masks = torch.zeros_like(input_ids) # 用于标记pad
            # for i, length in enumerate(original_lengths):
            #     attention_masks[i, :length] = 1
            batch = {'input_ids': input_ids, 'attention_mask': attention_masks}
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        labels = inputs.clone()

        # # 策略一、计算频率
        # range_1 = torch.sum((inputs > 31) & (inputs < 141)) # 4 15 for sz10; 31 141 for l96\144
        # range_2 = torch.sum(inputs <= 31)
        # range_3 = torch.sum(inputs >= 141)
        # total = range_1 + range_2 + range_3

        # # 根据tokens的频率设置MASK的概率
        # mask_prob_range_1 = 1 - range_1 / total
        # mask_prob_range_2 = 1 - range_2 / total
        # mask_prob_range_3 = 1 - range_3 / total

        # # 创建一个与输入相同的概率矩阵,并根据tokens的范围分配MASK概率
        # probability_matrix = torch.full(labels.shape, 1.0)
        # probability_matrix[(inputs > 31) & (inputs < 141)] *= mask_prob_range_1
        # probability_matrix[inputs <= 31] *= mask_prob_range_2
        # probability_matrix[inputs >= 141] *= mask_prob_range_3
        # # 将概率限制在min和max之间
        # probability_matrix_np = probability_matrix.numpy()
        # probability_matrix_np = np.clip(probability_matrix_np, a_min=0.30, a_max=0.60)
        # probability_matrix = torch.tensor(probability_matrix_np, dtype=torch.float)

        # 策略二、计算每个token的TF-IDF值
        token_list = labels.numpy().tolist()
        token_list = [[str(token) for token in sublist] for sublist in token_list]  # 将tokens转换为字符串
        token_list = [' '.join(sublist) for sublist in token_list]  # 将每个样本的tokens连接为一个字符串
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(token_list)
        tfidf = np.array(X.sum(axis=0)).ravel()  # 对每列求和，得到每个token的TF-IDF值
        tfidf = tfidf / tfidf.max()  # 归一化，将所有TF-IDF值都转化为0-1之间的值
        tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), tfidf))  # 创建一个字典，key是token，value是TF-IDF值

        # 根据每个token的TF-IDF值计算mask概率
        mask_prob = np.array([tfidf_dict.get(str(token), 0) for token in labels.numpy().ravel()])
        mask_prob = mask_prob.reshape(labels.shape)  # 调整形状，使之与输入的形状相同

        # 如果mask概率小于0.45,设置为0.45;如果高于0.6,设置为0.6
        mask_prob = np.clip(mask_prob, a_min=0.45, a_max=0.6)

        probability_matrix = torch.tensor(mask_prob, dtype=torch.float)
        probability_matrix[:, 0] = 0.0  # Ensure that the <cls> token is not masked


        # special_tokens_mask = torch.where(inputs==0, 1, 0).to(inputs.device)

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # 原始策略
        # probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # probability_matrix[:, 0] = 0.0  # Ensure that the <cls> token is not masked
        
        # 045-0.60随机mask
        # probability_matrix = torch.rand(labels.shape) * (0.60 - 0.45) + 0.45
        # probability_matrix[:, 0] = 0.0  # Ensure that the <cls> token is not masked

        if special_tokens_mask is None:
            # 取pad
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)


        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]


        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
