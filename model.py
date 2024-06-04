import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50,resnet101
import math
import timm

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)

    
    def forward(self, x, aux_imgs=None):
        # full image prompt 
        prompt_guids= self.get_resnet_prompt(x)    # 4x[bsz, 256, 7, 7]

        if aux_imgs is not None:
            aux_prompt_guids = []   # goal: 3 x (4 x [bsz, 256, 7, 7])
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # 3(nums) x bsz x 3 x 224 x 224
            for i in range(len(aux_imgs)):
                aux_prompt_guid= self.get_resnet_prompt(aux_imgs[i]) # 4 x [bsz, 256, 7, 7]
                aux_prompt_guids.append(aux_prompt_guid)

            return prompt_guids, aux_prompt_guids
        return prompt_guids, None



    def get_resnet_prompt(self, x):
         # The code will public after accept




class MoSiNetModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(MoSiNetREModel, self).__init__()
        self.bert = BertModel.from_pretrained('../bert-base-uncased')
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.resnet = resnet50(pretrained=True)
        self.tokenizer = tokenizer
        # the attention mechanism for fine-grained features
        self.hidden_size = 768 * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_pic = nn.Linear(2048, self.hidden_size // 2)
        self.linear_final = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.linear_q_fine = nn.Linear(768, self.hidden_size // 2)
        self.linear_k_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.conv1d = nn.Conv1d(in_channels=128, out_channels=49, kernel_size=1, stride=1)
        self.conv1d_t = nn.Conv1d(in_channels=20, out_channels=8, kernel_size=1, stride=1)
        self.linear_w = nn.Linear(768 * 2, 768)
        self.linear_t = nn.Linear(768, 8)
        self.linear_s = nn.Linear(1536, 768)
        self.linear_r = nn.Linear(8192, 6144)
        self.out_phrase = nn.Linear(768, num_labels)
        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_k_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        # the attention mechanism for entity features
        self.linear_q_entity = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.linear_k_entity = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.linear_v_entity = nn.Linear(self.hidden_size, self.hidden_size // 2)

        self.linear_weights = nn.Linear(self.hidden_size * 3, 3)
        self.dropout_linear = nn.Dropout(0.5)
        self.weights = nn.Parameter(torch.tensor([0.0, 1.0]))
        self.image_model = ImageModel()
        self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                )
        self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        input_ids_phrase=None,
        attention_mask_phrase=None,
        labels=None,
        images=None,
        aux_imgs=None,

    ):
    # The code will public after accept


    def att(self, query, key, value):
        # The code will public after accept

        return torch.matmul(att_map, value)
    def get_resnet_feature(self, x):
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
        return x

    def get_visual_prompt(self, images, aux_imgs, hybrid):
       # The code will public after accept

    def softmax(self, x, dim):
        # The code will public after accept






class HMNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained('../bert-base-uncased')
        self.bert_config = self.bert.config
        self.image2token_emb = 1024
        self.resnet = resnet101(pretrained=True)  # get the pre-trained ResNet model for the image

        self.num_labels  = len(label_list)  # pad
        print(self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(1000, self.num_labels)
        self.dropout = nn.Dropout(0.5)

        self.linear_extend_pic = nn.Linear(self.bert.config.hidden_size, self.args.max_seq * self.image2token_emb)
        self.linear_pic = nn.Linear(2048, self.bert.config.hidden_size)

        # the attention mechanism for fine-grained features
        self.linear_q_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_k_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_v_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_k_coarse = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_v_coarse = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        self.combine_linear = nn.Linear(self.bert.config.hidden_size + self.image2token_emb, 1000)
        self.image_model = ImageModel()
        self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                )
        self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None,
                weight=None, input_ids_phrase=None, attention_mask_phrase=None):
       # The code will public after accept

    
    def get_resnet_feature(self, x):
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
        return x
    def att(self, query, key, value):
        # The code will public after accept

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(sum=0.0, std=0.05)
    def get_visual_prompt(self, images, aux_imgs):
        # The code will public after accept

    def softmax(self, x, dim):
        # The code will public after accept
