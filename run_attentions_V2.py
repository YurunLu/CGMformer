# coding: utf-8
# imports
import logging
import pickle
import seaborn as sns;
import numpy as np
import os
# 会让CUDA内核(kernels)阻塞CPU执行,在开发和调试CUDA代码时让程序的行为更加地确定性和易理解
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0" 
# 设置 matplotlib 的样式
sns.set()
import torch
import torch.nn.functional as F
import pickle
from datasets import Dataset, load_from_disk
from transformers import BertForMaskedLM
import matplotlib.pyplot as plt
import debugpy
from tqdm import tqdm

from bertviz import head_view, model_view

from CGMFormer import BertForSequenceClassification
from CGMFormer import DataCollatorForCellClassification
from CGMFormer import ClasssifyTrainer

from IPython.display import display, HTML, Javascript

# debugpy.listen(("192.168.72.59", 5681)) # g04
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'


def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums)) + 1
logger = logging.getLogger(__name__)

class AttentionVisualization:
    def __init__(
            self,
            checkPointFile,
            visualization_dataset,
            load_preTrain=True,
            token_dictionary_file=TOKEN_DICTIONARY_FILE,
            emb_layer=-1,
            nproc=16,
    ):
        self.checkPointFile = checkPointFile
        self.visualization_dataset = visualization_dataset
        self.nproc = nproc
        self.emb_layer = emb_layer
        self.load_preTrain = load_preTrain

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.token_dictionary = pickle.load(f)


    def perturb_data_IPSCs(self, model_directory):
        # filtered_input_data = self.load_and_filter(input_data_file)
        with open('/share/home/liangzhongming/930/xCompass_628_1/xCompass/pertub_patch/fibpatch1.pickle','rb') as f:
            fib = pickle.load(f)
        with open('/share/home/liangzhongming/930/xCompass_628_1/xCompass/pertub_patch/IPSCpatch1.pickle','rb') as f:
            ipsc = pickle.load(f)
        ipsc = torch.tensor(ipsc) # torch.Size([2, 2048])
        model = self.load_model(model_directory)
        perturb_token = np.array(self.tokens_to_perturb) #需要扰动的token
        layer_to_quant = quant_layers(model) + self.emb_layer # 3
        perturb_parameters = 20
        #1.将扰动的token加上认为设置的values加到原有数据的前三位，并挤掉后面三位
        max_value = fib[1][0] # value 148.03597194001912
        max_value = np.array([max_value + perturb_parameters] * len(self.tokens_to_perturb))
        perturb = np.vstack((perturb_token,max_value))
        fib[:, -perturb.shape[1]:] = perturb #替换最后几位
        per_input = torch.tensor(fib[:, np.argsort(-fib[1])]) #扰动的输入
        #2.计算扰动后成纤细胞与ipsc的Embedding
        with torch.no_grad():
            per_emb = model(per_input.permute(1,0).long().to(device)) # 2*2048 ——> 2048*2
        per_out = torch.squeeze(per_emb.hidden_states[layer_to_quant]) # torch.Size([2048, 2, 128])
        per_out = per_out.reshape(2048,-1) #扰动嵌入

        with torch.no_grad():
            ipsc_emb = model(ipsc.permute(1,0).long().to(device))
        ipsc_out = torch.squeeze(ipsc_emb.hidden_states[layer_to_quant])
        ipsc_out = ipsc_out.reshape(2048,-1) #扰动嵌入
         #3.计算重叠基因
        # ipsc_token = []
        # for i in ipsc[0]:
        #     ipsc_token.append(int(i))
        # fib_token = []
        # for i in fib[0]:
        #     fib_token.append(int(i))
        # ipsc_token = set(ipsc_token)
        # fib_token = set(fib_token)
        # intersection = ipsc_token.intersection(fib_token)
        # fib_itersec = []
        # for i, data in enumerate(fib[0]):
        #     if int(data) in intersection:
        #         fib_itersec.append(i)
        # ipsc_itersec = []
        # for i, data in enumerate(ipsc[0]):
        #     if int(data) in intersection:
        #         ipsc_itersec.append(i)
        # result_fib = [per_out[index] for index in fib_itersec]
        # result_ipsc = [ipsc_out[index] for index in ipsc_itersec]
        # result_fib = torch.stack(result_fib)
        # result_ipsc = torch.stack(result_ipsc)
        #3.计算余弦相似度
        cos = torch.nn.CosineSimilarity(dim=1)
        out_save = cos(per_out,ipsc_out).tolist()
        #4.存储pickle
        file_handle = open('/share/home/liangzhongming/930/xCompass_628_1/xCompass/down_stream_outputs/cossim.pickle', 'wb')
        # 使用 pickle.dump() 存储对象到文件
        pickle.dump(out_save, file_handle)
        # 关闭文件
        file_handle.close()

    # load model to GPU
    def load_model(self):
        if self.load_preTrain:
            # 加载预训练模型
            model = BertForMaskedLM.from_pretrained(self.checkPointFile,
                                                    output_hidden_states=True,
                                                    output_attentions=True)

        else:
        # 加载分类模型
            model = BertForSequenceClassification.from_pretrained(
                self.checkPointFile,
                # num_labels=len(subtask_label_dict.keys()), # Colas数据集 + 1
                output_attentions=True,
                output_hidden_states=False, # False->True
                # ignore_mismatched_sizes=True, # 忽略参数大小不匹配的情况
            )
        # put the model in eval mode for fwd pass
        model.eval()
        model = model.to(device)
        return model

    def tokenizer(self, totalData):
        # totalData = load_from_disk(self.visualization_dataset)
              
        
        cls_token_id = self.token_dictionary['<cls>'] # 265
        pad_token_id = self.token_dictionary['<pad>'] # 264
        
        input_ids = []
        for seq in totalData:
            # tokens = np.array(seq['input_ids'], dtype=float)
            tokens = np.array(seq, dtype=float)
            nan_index = np.isnan(tokens)
            
            tokens[~nan_index] = np.clip(tokens[~nan_index], 39, 301) 
            tokens[~nan_index] = [self.token_dictionary[int(token)] for token in tokens[~nan_index]]
            
            tokens[nan_index] = pad_token_id 
            tokens_with_cls = np.insert(tokens, 0, cls_token_id)

            input_ids.append(tokens_with_cls.tolist())

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids


    def run(self):

        whole_data = load_from_disk(self.visualization_dataset)
        # 1440, 1272, 1100, 1073, 621, 198, 927, 369, 636, 92, 172, 1166, 877, 1051
        # index_to_extract = 1051
        # index_list = [i for i, idx in enumerate(whole_data['index']) if idx == index_to_extract]
        # if index_list:
        #     origined_sequence = [whole_data['input_ids'][i] for i in index_list]
        
        origined_sequence = whole_data['input_ids']
        # int_Data = [torch.tensor(seq, dtype=torch.long) for seq in origined_sequence]

        processed_data = []
        for sequence in origined_sequence:
            sequence = [str(int(token)) for token in sequence]
            sequence = ['cls'] + sequence
            processed_data.append(sequence)

        # # 处理 NaN of Colas
        # for i, sequence in enumerate(origined_sequence):
        #     for j, token in enumerate(sequence):
        #         if np.isnan(token):
        #             continue
        #             print(f"NaN found at index [{i},{j}]")
        #         else: 
        #             token = str(int(token))
        #     sequence = ['cls'] + sequence
        #     processed_data.append(sequence)
        
        # 用于bertviz的可视化,原始序列
        processed_data_flat = [str(token) for sequence in processed_data for token in sequence]


        data = torch.tensor(self.tokenizer(origined_sequence)) #
        # batch1 = data[:300]
        # batch2 = data[300:600]
        # batch3 = data[600:900]
        # batch4 = data[900:1200]
        # batch5 = data[1200:1500]
        batch6 = data[1500:]
        
        model = self.load_model()

        layer_to_quant = quant_layers(model) + self.emb_layer # 3

        with torch.no_grad():
            # per_emb = model(per_input.permute(1,0).long().to(device))
            # per_emb = model(ipsc.permute(1,0).long().to(device))
            # outputs = model(data.to(device)).attentions

            # attentions_batch1 = model(batch1.to(device)).attentions
            # attentions_batch2 = model(batch2.to(device)).attentions
            # attentions_batch3 = model(batch3.to(device)).attentions
            # attentions_batch4 = model(batch4.to(device)).attentions
            # attentions_batch5 = model(batch5.to(device)).attentions
            attentions_batch6 = model(batch6.to(device)).attentions

        # merged_attentions = torch.cat([attentions_batch1, attentions_batch2, attentions_batch3], dim=0)
        # with open(f"/share/home/liangzhongming/930/CGMformer/output/outputs_attentions/seq{index_to_extract}_attention.pkl", 'wb') as f:
        #     pickle.dump(attentions, f)

        # outputs = tuple(att.cpu() for att in outputs)
        # with open(f"/share/home/liangzhongming/930/CGMformer/output/outputs_attentions/shanghai_total_finetuneAttention.pkl", 'wb') as f:
        #     pickle.dump(outputs, f)

        attentions_batch6 = tuple(att.cpu() for att in attentions_batch6)
        with open(f"/share/home/liangzhongming/930/CGMformer/output/outputs_attentions/shanghai_batch6_finetuneAttention.pkl", 'wb') as f:
            pickle.dump(attentions_batch6, f)



        # html_model = model_view(attentions, processed_data_flat, html_action='return')
        # html_head = head_view(attentions, processed_data_flat, html_action='return')
        # with open("/share/home/liangzhongming/930/CGMformer/output/high_low_96bert_model.html", 'w') as file:
        #     file.write(html_model.data)
        # with open("/share/home/liangzhongming/930/CGMformer/output/high_low_96bert_head.html", 'w') as file:
        #     file.write(html_head.data)


if __name__ == '__main__':
    # 80w preTrain
    # checkPointFile = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
    
    # checkPointFile = "/share/home/liangzhongming/930/geneformer_ia-master-11f0c55feb5a7d87916284c2187f0a1c6c4afb4f/output_dir/models/230702_001335_h&m&m_50WP_L4_emb128_SL2048_E3_B4_LR0.001_LSlinear_WU10000_Oadamw_DS2/checkpoint-470000"
    # visualization_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downstream/288/CV_1/test"
    
    # 96
    # checkPointFile = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230820_013424_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL97_E3000_B48_LR0.0004_LScosine_WU2000_Oadamw_DS2/checkpoint-30000"
    # visualization_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/downsampled_CV_1_test_96"
    

    # 288
    # checkPointFile = "/share/home/liangzhongming/930/CGMformer/output/output_dir813/models/230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
    # visualization_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_total"

    checkPointFile = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231028_125607_dim128_linear_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
    visualization_dataset = "/share/home/liangzhongming/930/CGMformer/data/8_11_data/Shanghai_total"
    # visualization_dataset = "/share/home/liangzhongming/930/CGMformer/data/Data_downstream/Colas_1028_hfds"

    # 分类器checkpoint
    classify_checkpoint_path = "/share/home/liangzhongming/930/CGMformer/downStreamOutput/819/288_Shanghai_CV1_F0_230907_230818_141954_TFIDF4560_sincos_SZ1_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2_L289_B12_LR0.00041_LScosine_WU1385_E20_Oadamw_F0/checkpoint-546"


    # LOAD_PRETRAIN = True
    
    visit = AttentionVisualization(
                            classify_checkpoint_path,
                            visualization_dataset,
                            emb_layer=-1,
                            nproc=16,
                            # LOAD_PRETRAIN = True,
                            )
    visit.run()
    
    print("done!")