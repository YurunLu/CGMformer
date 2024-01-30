
# imports
import logging
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scanpy as sc
import seaborn as sns
import torch
from collections import Counter
from pathlib import Path
from tqdm.notebook import trange
from transformers import BertForMaskedLM, BertForTokenClassification, BertForSequenceClassification

# from .tokenizer import TOKEN_DICTIONARY_FILE

from .utils import downsample_and_sort, \
                                 gen_attention_mask, \
                                 get_model_input_size, \
                                 load_and_filter, \
                                 load_model, \
                                 mean_nonpadding_embs, \
                                 pad_tensor_list, \
                                 quant_layers

# TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_8_data/288/Shanghai_total_token2id.pkl'
TOKEN_DICTIONARY_FILE = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl"
# TOKEN_DICTIONARY_FILE = "/share/home/liangzhongming/930/CGMformer/data/8_2_newData/Shanghai_vocab_rank.pkl"
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dictionary = pickle.load(f)

logger = logging.getLogger(__name__)

# average embedding position of goal cell states
def get_embs(model,
             max_length,
             filtered_input_data,
             emb_mode,
             layer_to_quant,
             pad_token_id,
             forward_batch_size):
    
    model_input_size = get_model_input_size(model)
    total_batch_length = len(filtered_input_data)
    if ((total_batch_length-1)/forward_batch_size).is_integer():
        forward_batch_size = forward_batch_size-1
    

    embs_list = []
    for i in trange(0, total_batch_length, forward_batch_size):
        max_range = min(i+forward_batch_size, total_batch_length)

        minibatch = filtered_input_data.select([i for i in range(i, max_range)])
        # max_len = max(minibatch["length"])
        max_len = max_length
        # max_len = 96
        # original_lens = torch.tensor(minibatch["length"]).to("cuda")
        original_lens = torch.tensor(max_length).to("cuda")
        minibatch.set_format(type="torch")

        cls_token_id = token_dictionary['<cls>']
        pad_token_id = token_dictionary['<pad>']
        input_data_minibatch = []

        # for example in minibatch["input_ids"]:
        #     tokens = np.array(example, dtype=int)
        #     tokens[tokens < 40] = 39
        #     tokens[tokens > 300] = 301
        #     tokens_with_cls = np.insert(tokens, 0, cls_token_id)
        #     input_data_minibatch.append([token_dictionary[token] for token in tokens_with_cls])
        
        for example in minibatch["input_ids"]:
            tokens = np.array(example, dtype=float)
            nan_index = np.isnan(tokens)
            
            tokens[~nan_index] = np.clip(tokens[~nan_index], 39, 301) 
            tokens[~nan_index] = [token_dictionary[int(token)] for token in tokens[~nan_index]]
            
            tokens[nan_index] = pad_token_id 
            tokens_with_cls = np.insert(tokens, 0, cls_token_id)
            
            tokens_with_cls = np.insert(tokens, 0, cls_token_id)
            # tokens_with_cls = tokens_with_cls.astype(int)
            input_data_minibatch.append(tokens_with_cls)

        input_data_minibatch = torch.tensor(input_data_minibatch, dtype=torch.long)

        input_data_minibatch = pad_tensor_list(input_data_minibatch, 
                                               max_len, 
                                               pad_token_id, 
                                               model_input_size)

        with torch.no_grad():
            outputs = model(
                input_ids = input_data_minibatch.to("cuda"),
                attention_mask = gen_attention_mask(minibatch, max_len=max_length)
            )
        
        embs_i = outputs.hidden_states[layer_to_quant]
        
        if emb_mode == "sample":
            mean_embs = mean_nonpadding_embs(embs_i, original_lens)
            embs_list += [mean_embs]
            # first_embs = embs_i[:, 0, :]
            # embs_list.append(first_embs)
            
        del outputs
        del minibatch
        del input_data_minibatch
        del embs_i
        del mean_embs
        # del first_embs
        torch.cuda.empty_cache()
        
    embs_stack = torch.cat(embs_list)
    return embs_stack

def get_range_embs(model,
             max_length,
             filtered_input_data,
             emb_mode,
             layer_to_quant,
             pad_token_id,
             forward_batch_size):
    
    model_input_size = get_model_input_size(model)
    total_batch_length = len(filtered_input_data)
    if ((total_batch_length-1)/forward_batch_size).is_integer():
        forward_batch_size = forward_batch_size-1
    
    embs_list = []
    for i in trange(0, total_batch_length, forward_batch_size):
        max_range = min(i+forward_batch_size, total_batch_length)

        minibatch = filtered_input_data.select([i for i in range(i, max_range)])
        # max_len = max(minibatch["length"])
        # original_lens = torch.tensor(minibatch["length"]).to("cuda")
        original_lens = torch.tensor(max_length).to("cuda")
        minibatch.set_format(type="torch")

        # input_data_minibatch = torch.tensor([[token_dictionary[(int(token) - 30) // 10] for token in example] for example in minibatch["input_ids"]], dtype=torch.long)
        
        cls_token_id = token_dictionary['<cls>']
        input_data_minibatch = []

        for example in minibatch["input_ids"]:
            tokens = np.array(example, dtype=int)
            tokens[tokens < 40] = 39
            tokens[tokens > 300] = 301
            tokens_with_cls = np.insert(tokens, 0, cls_token_id) 
            input_data_minibatch.append([token_dictionary[token] for token in tokens_with_cls])

        input_data_minibatch = torch.tensor(input_data_minibatch, dtype=torch.long)    
        
        # input_data_minibatch = torch.tensor([[token_dictionary[int(token)] for token in example] for example in minibatch["input_ids"]], dtype=torch.long)
        # input_data_minibatch2 = minibatch["input_ids"]
        input_data_minibatch = pad_tensor_list(input_data_minibatch, 
                                               max_length, 
                                               pad_token_id, 
                                               model_input_size)
        
        with torch.no_grad():
            outputs = model(
                input_ids = input_data_minibatch.to("cuda"),
                attention_mask = gen_attention_mask(minibatch, max_len=max_length)
            )
        
        embs_i = outputs.hidden_states[layer_to_quant]

        stage = "Fast"
        
        # Fast Stage Embedding
        # fast_embs_dict = {i: [] for i in range(len(embs_i))}
        for i, sample in enumerate(embs_i):

            start_idx = int(minibatch[f'{stage}_s'][i].item()) + 1
            end_idx = int(minibatch[f'{stage}_e'][i].item()) + 1
            stage_embs = sample[start_idx:end_idx, :]
            stage_mean_embs = [torch.mean(stage_embs, dim=0)]
            # fast_embs_dict[i].append(stage_mean_embs)
            embs_list += stage_mean_embs

        
        # for i in fast_embs_dict:
        #     fast_embs_dict[i] = torch.stack(fast_embs_dict[i])
    
    embs_stack = torch.cat(embs_list)
    return embs_list

def label_embs(embs, downsampled_data, emb_labels):
    # embs_df = pd.DataFrame(embs.cpu())
    embs = [emb.cpu() for emb in embs] 

    embs = [emb.detach().numpy() for emb in embs]
    embs_df = pd.DataFrame(embs)
    if emb_labels is not None:
        for label in emb_labels:
            emb_label = downsampled_data[label]
            embs_df[label] = emb_label
    return embs_df

def label_range_embs(embs_dict, downsampled_data, emb_labels):
    embs_dict_df = {}
    
    for stage_name in embs_dict:
        stage_embs = embs_dict[stage_name]
        stage_embs_df = pd.DataFrame(stage_embs.cpu())
        if emb_labels is not None:
            for label in emb_labels:
                emb_label = downsampled_data[label]
                stage_embs_df[label] = emb_label
        embs_dict_df[stage_name] = stage_embs_df
        
    return embs_dict_df

def plot_umap(embs_df, emb_dims, label, output_file, kwargs_dict):
    only_embs_df = embs_df.iloc[:,:emb_dims]
    only_embs_df.index = pd.RangeIndex(0, only_embs_df.shape[0], name=None).astype(str)
    only_embs_df.columns = pd.RangeIndex(0, only_embs_df.shape[1], name=None).astype(str)
    vars_dict = {"embs": only_embs_df.columns}
    obs_dict = {"sample_id": list(only_embs_df.index),
                f"{label}": list(embs_df[label])}
    adata = anndata.AnnData(X=only_embs_df, obs=obs_dict, var=vars_dict)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sns.set(rc={'figure.figsize':(10,10)}, font_scale=2.3)
    sns.set_style("white")
    default_kwargs_dict = {"palette":"Set2", "size":200}
    if kwargs_dict is not None:
        default_kwargs_dict.update(kwargs_dict)
        
    sc.pl.umap(adata, color=label, save=output_file, **default_kwargs_dict)
 
def gen_heatmap_class_colors(labels, df):
    pal = sns.cubehelix_palette(len(Counter(labels).keys()), light=0.9, dark=0.1, hue=1, reverse=True, start=1, rot=-2)
    lut = dict(zip(map(str, Counter(labels).keys()), pal))
    colors = pd.Series(labels, index=df.index).map(lut)
    return colors
    
def gen_heatmap_class_dict(classes, label_colors_series):
    class_color_dict_df = pd.DataFrame({"classes": classes, "color": label_colors_series})
    class_color_dict_df = class_color_dict_df.drop_duplicates(subset=["classes"])
    return dict(zip(class_color_dict_df["classes"],class_color_dict_df["color"]))
    
def make_colorbar(embs_df, label):

    labels = list(embs_df[label])
                  
    cell_type_colors = gen_heatmap_class_colors(labels, embs_df)
    label_colors = pd.DataFrame(cell_type_colors, columns=[label])

    for i,row in label_colors.iterrows():
        colors=row[0]
        if len(colors)!=3 or any(np.isnan(colors)):
            print(i,colors)

    label_colors.isna().sum()
    
    # create dictionary for colors and classes
    label_color_dict = gen_heatmap_class_dict(labels, label_colors[label])
    return label_colors, label_color_dict
    
def plot_heatmap(embs_df, emb_dims, label, output_file, kwargs_dict):
    sns.set_style("white")
    sns.set(font_scale=2)
    plt.figure(figsize=(15, 15), dpi=150)
    label_colors, label_color_dict = make_colorbar(embs_df, label)
    
    default_kwargs_dict = {"row_cluster": True,
                           "col_cluster": True,
                           "row_colors": label_colors,
                           "standard_scale":  1,
                           "linewidths": 0,
                           "xticklabels": False,
                           "yticklabels": False,
                           "figsize": (15,15),
                           "center": 0,
                           "cmap": "magma"}
    
    if kwargs_dict is not None:
        default_kwargs_dict.update(kwargs_dict)
    g = sns.clustermap(embs_df.iloc[:,0:emb_dims].apply(pd.to_numeric), **default_kwargs_dict)

    plt.setp(g.ax_row_colors.get_xmajorticklabels(), rotation=45, ha="right")

    for label_color in list(label_color_dict.keys()):
        g.ax_col_dendrogram.bar(0, 0, color=label_color_dict[label_color], label=label_color, linewidth=0)

        l1 = g.ax_col_dendrogram.legend(title=f"{label}", 
                                        loc="lower center", 
                                        ncol=4, 
                                        bbox_to_anchor=(0.5, 1), 
                                        facecolor="white")

    plt.savefig(output_file, bbox_inches='tight')

class EmbExtractor:
    valid_option_dict = {
        "model_type": {"Pretrained","TokenClassifier","SampleClassifier"},
        "max_length": {int},
        "num_classes": {int},
        "emb_mode": {"sample","token"},
        "sample_emb_style": {"mean_pool", "cls"},
        "filter_data": {None, dict},
        "filter_target":{'types' ,'Fast_s', 'Fast_e', 'Dawn_s', 'Dawn_e', 'Breakfast_s', 'Breakfast_e', 'Lunch_s', 'Lunch_e', 'Dinner_s', 'Dinner_e', 'type', 'cerebrovascular disease', 'coronary heart disease', 'peripheral arterial disease', 'nephropathy', 'neuropathy', 'retinopathy', 'macrovascular', 'microvascular', 'complication'},
        "max_nsamples": {None, int},
        "emb_layer": {-1, 0},
        "emb_label": {None, list},
        "labels_to_plot": {None, list},
        "forward_batch_size": {int},
        "nproc": {int},
    }
    def __init__(
        self,
        model_type="SampleClassifier",
        max_length=0,
        num_classes=0,
        emb_mode="sample",
        sample_emb_style="mean_pool",
        filter_data=None,
        filter_target=None,
        max_nsamples=2000,
        emb_layer=-1,
        emb_label=None,
        labels_to_plot=None,
        forward_batch_size=100,
        nproc=4,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        
        self.model_type = model_type
        self.max_length = max_length
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.sample_emb_style = sample_emb_style
        self.filter_data = filter_data
        self.filter_target = filter_target
        self.max_nsamples = max_nsamples
        self.emb_layer = emb_layer
        self.emb_label = emb_label
        self.labels_to_plot = labels_to_plot
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc

        self.validate_options()

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self.pad_token_id = self.gene_token_dict.get("<pad>")
        
        
    def validate_options(self):
        # first disallow options under development
        if self.emb_mode == "token":
            logger.error(
                "Extraction and plotting of token-level embeddings currently under development. " \
                "Current valid option for 'emb_mode': 'sample'"
            )
            raise
            
        # confirm arguments are within valid options and compatible with each other
        for attr_name,valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [int,list,dict]) and isinstance(attr_value, option):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. " \
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise
        
        if self.filter_data is not None:
            for key,value in self.filter_data.items():
                if type(value) != list:
                    self.filter_data[key] = [value]
                    logger.warning(
                        "Values in filter_data dict must be lists. " \
                        f"Changing {key} value to list ([{value}]).")  
        
    def extract_embs(self, 
                     model_directory,
                     input_data_file,
                     output_directory,
                     output_prefix):
        """
        Extract embeddings from input data and save as results in output_directory.
        Parameters
        ----------
        model_directory : Path
            Path to directory containing model
        input_data_file : Path
            Path to directory containing .dataset inputs
        output_directory : Path
            Path to directory where embedding data will be saved as csv
        output_prefix : str
            Prefix for output file
        """

        filtered_input_data = load_and_filter(self.filter_data, self.filter_target, self.nproc, input_data_file)
        downsampled_data = downsample_and_sort(filtered_input_data, self.max_nsamples)
        model = load_model(self.model_type, self.num_classes, model_directory)
        layer_to_quant = quant_layers(model)+self.emb_layer
        embs = get_embs(model,
                        self.max_length,
                        downsampled_data,
                        self.emb_mode,
                        layer_to_quant,
                        self.pad_token_id,
                        self.forward_batch_size)
        # range_embs_dict = get_range_embs(model,
        #                                 self.max_length,
        #                                 downsampled_data,
        #                                 self.emb_mode,
        #                                 layer_to_quant,
        #                                 self.pad_token_id,
        #                                 self.forward_batch_size)

        embs_df = label_embs(embs, downsampled_data, self.emb_label)

        # embs_df = label_range_embs(range_embs_dict, downsampled_data, self.emb_label)
        
        # save embeddings to output_path
        output_path1_embs = (Path(output_directory) / output_prefix).with_suffix(".csv")
        output_path2_embs = (Path(output_directory) / output_prefix).with_suffix(".npy")
        embs_df.to_csv(output_path1_embs)
        np.save(output_path2_embs, embs_df.values)

        # output_path1_embs_dict = (Path(output_directory) / output_prefix + "_embs_dict").with_suffix(".csv")
        # output_path2_embs_dict = (Path(output_directory) / output_prefix + "_embs_dict").with_suffix(".npy")
        # embs_df.to_csv(output_path1_embs_dict)
        # np.save(output_path2_embs_dict, embs_df.values)
        
        return embs_df
    


    def plot_embs(self,
                  embs, 
                  plot_style,
                  output_directory,
                  output_prefix,
                  max_nsamples_to_plot=2000,
                  kwargs_dict=None):
        
        
        if plot_style not in ["heatmap","umap"]:
            logger.error(
                "Invalid option for 'plot_style'. " \
                "Valid options: {'heatmap','umap'}"
            )
            raise
        
        if (plot_style == "umap") and (self.labels_to_plot is None):
            logger.error(
                "Plotting UMAP requires 'labels_to_plot'. "
            )
            raise
        
        if max_nsamples_to_plot > self.max_nsamples:
            max_nsamples_to_plot = self.max_nsamples
            logger.warning(
                "max_nsamples_to_plot must be <= max_nsamples. " \
                f"Changing max_nsamples_to_plot to {self.max_nsamples}.") 
        
        if (max_nsamples_to_plot is not None) \
            and (max_nsamples_to_plot < self.max_nsamples):
            embs = embs.sample(max_nsamples_to_plot, axis=0)
        
        if self.emb_label is None:
            label_len = 0
        else:
            label_len = len(self.emb_label)
        
        emb_dims = embs.shape[1] - label_len
        
        if self.emb_label is None:
            emb_labels = None
        else:
            emb_labels = embs.columns[emb_dims:]
        
        if plot_style == "umap":
            for label in self.labels_to_plot:
                if label not in emb_labels:
                    logger.warning(
                        f"Label {label} from labels_to_plot " \
                        f"not present in provided embeddings dataframe.")
                    continue
                output_prefix_label = "_" + output_prefix + f"_umap_{label}"
                output_file = (Path(output_directory) / output_prefix_label).with_suffix(".pdf")
                plot_umap(embs, emb_dims, label, output_prefix_label, kwargs_dict)
                
        if plot_style == "heatmap":
            for label in self.labels_to_plot:
                if label not in emb_labels:
                    logger.warning(
                        f"Label {label} from labels_to_plot " \
                        f"not present in provided embeddings dataframe.")
                    continue
                output_prefix_label = output_prefix + f"_heatmap_{label}"
                output_file = (Path(output_directory) / output_prefix_label).with_suffix(".pdf")
                plot_heatmap(embs, emb_dims, label, output_file, kwargs_dict)
                


