
# import pickle
# import torch
# from transformers import BertConfig, BertForMaskedLM

# token_dict_path = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'

# def calculate_model_parameters(model):
#     total_params = 0
#     transformer_params = 0

#     for name, parameter in model.named_parameters():
#         total_params += parameter.numel()
        
#         if 'transformer' in name:
#             transformer_params += parameter.numel()

#     print(f"Total Number of Parameters: {total_params:,}")
#     print(f"Transformer Layers Number of Parameters: {transformer_params:,}")


# with open(token_dict_path, "rb") as fp:
#     token_dictionary = pickle.load(fp)

# model_type = "bert"
# # max input size
# max_input_size = 289 # <cls> + 96/ <cls> + 288
# # number of layers
# num_layers = 4
# # number of attention heads
# num_attn_heads = 8 # 8——>4
# # number of embedding dimensions
# num_embed_dim = 128
# # intermediate size
# intermed_size = num_embed_dim * 4 # num_embed_dim * 4——>num_embed_dim * 2
# # activation function
# activ_fn = "gelu" # relu->gelu
# # initializer range, layer norm, dropout
# initializer_range = 0.02 # Bert default 0.02
# layer_norm_eps = 1e-12 # 
# attention_probs_dropout_prob = 0.02 # Bert default 0.1 # 0.02->0.1
# hidden_dropout_prob = 0.02 # Bert default 0.1 # 0.02->0.1 

# # set training parameters
# # number gpus
# num_gpus = 2
# # batch size for training and eval
# batch_size = 48 
# # max learning rate
# max_lr = 4e-4 # 8e-6(2000epoch) 4e-6 best
# # learning schedule
# lr_schedule_fn = "linear" # linear->cossin
# # warmup steps
# warmup_steps = 2000 
# # number of epochs
# epochs = 3000 
# # optimizer
# optimizer = "adamw"
# # weight_decay
# weight_decay = 0.001 # 0.001->0.01


# # config = BertConfig(
# #     hidden_size=128,
# #     num_hidden_layers=4,
# #     num_attention_heads=8,
# #     intermediate_size=512
# # )

# config = {
#     "hidden_size": num_embed_dim,
#     "num_hidden_layers": num_layers,
#     "initializer_range": initializer_range,
#     "layer_norm_eps": layer_norm_eps,
#     "attention_probs_dropout_prob": attention_probs_dropout_prob,
#     "hidden_dropout_prob": hidden_dropout_prob,
#     "intermediate_size": intermed_size,
#     "hidden_act": activ_fn,
#     "max_position_embeddings": max_input_size,
#     "model_type": model_type,
#     "num_attention_heads": num_attn_heads,
#     "pad_token_id": token_dictionary.get("<pad>"),
#     "vocab_size": len(token_dictionary),
#     # "use_cls_token": use_cls_token,

# }

# config = BertConfig(**config)

# model = BertForMaskedLM(config)

# calculate_model_parameters(model)


from transformers import BertForSequenceClassification

checkpoint_1917_path = "/share/home/liangzhongming/930/CGMformer/output/output_ablation/models/231028_125607_dim128_linear_L4_H8_emb128_SL289_E3000_B48_LR0.0004_LSlinear_WU2000_Oadamw_DS2/checkpoint-30000"
subtask_label_dict = {'label1': 0, 'label2': 1} 
model = BertForSequenceClassification.from_pretrained(
    checkpoint_1917_path,
    num_labels=len(subtask_label_dict.keys()),
    output_attentions=False,
    output_hidden_states=False, # False->True
)

total_params = 0
transformer_params = 0

for name, parameter in model.named_parameters():
    total_params += parameter.numel()
    
    if 'bert' in name:
        transformer_params += parameter.numel()

print(f"Total Number of Parameters: {total_params:,}")
print(f"Transformer Layers Number of Parameters: {transformer_params:,}")
