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
        """generate image prompt

        Args:
            x ([torch.tenspr]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 7 x 7]
        """
        # image: bsz x 3 x 224 x 224
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)    # (bsz, 256, 56, 56)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size() #256是channel，56是feature map的大小
                kernel = ft // 2 # 28
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)    # (bsz, 256, 7, 7)
                prompt_guids.append(prompt_kv)   # conv2: (bsz, 256, 7, 7)

        return prompt_guids




class MoSiNetModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(MoSiNetREModel, self).__init__()
        #self.bert = BertModel.from_pretrained(args.bert_name)
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

        if self.args.use_prompt:
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
        bsz = input_ids.size(0)
        # modify by zhang11.26
        output_phrases = self.bert(input_ids_phrase, attention_mask=attention_mask_phrase)
        hidden_phrases = output_phrases[0]
        hybrid = hidden_phrases
        hybrid = self.conv1d_t(hybrid)
        hybrid = hybrid.view(bsz, -1, 768)
        if self.args.use_prompt:
            #modify by zhang11.7
            prompt_guids = self.get_visual_prompt(images, aux_imgs, hybrid)#(12, 2, 16, 12, 4, 64)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        image_ori = images
        image_ori_objects = aux_imgs.reshape(-1, 3, 224, 224)
        feature_OriImg_FineGrained = self.get_resnet_feature(image_ori)
        feature_OriImg_CoarseGrained = self.get_resnet_feature(image_ori_objects)

        pic_ori = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori = torch.transpose(pic_ori, 1, 2)
        pic_ori = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori = self.linear_pic(pic_ori)

        pic_ori_objects = torch.reshape(feature_OriImg_CoarseGrained, (-1, 2048, 49))
        pic_ori_objects = torch.transpose(pic_ori_objects, 1, 2)
        pic_ori_objects = torch.reshape(pic_ori_objects, (-1, 3, 49, 2048))
        pic_ori_objects = torch.sum(pic_ori_objects, dim=2)
        pic_ori_objects = self.linear_pic(pic_ori_objects)

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden

        hidden_k_text = self.linear_k_fine(last_hidden_state)
        hidden_v_text = self.linear_v_fine(last_hidden_state)
        pic_q_origin = self.linear_q_fine(pic_ori)

        temp_T = self.conv1d(hidden_v_text)
        temp_T = temp_T.view(bsz, 49, 768)
        pic_q_origin_1 = torch.cat((pic_q_origin, temp_T), dim=-1)
        pic_q_origin_1 = self.linear_w(pic_q_origin_1)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin_1, hidden_k_text, hidden_v_text)), dim=1)


        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_ori_objects = self.linear_q_coarse(pic_ori_objects)
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)),
                                         dim=1)

        hidden_phrases = torch.sum(hidden_phrases, dim=1)

        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

        entity_hidden_state = entity_hidden_state.to(self.args.device)

        x = torch.cat([hidden_phrases, entity_hidden_state,
                       pic_original+pic_original_objects], dim=-1)
        x = self.linear_final(self.dropout_linear(x))
        x = self.dropout(x)

        logits = self.classifier(x)
        logits_phrase = self.out_phrase(hidden_phrases)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)) + 0.5*loss_fn(logits_phrase, labels.view(-1)), logits, x
        return logits, x


    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)
    def get_resnet_feature(self, x):
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
        return x

    def get_visual_prompt(self, images, aux_imgs, hybrid):
        bsz = images.size(0)
        # full image prompt
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        # Hybrid Modal Network
        hybrid = hybrid.view(bsz, -1)
        last_resnet_layer = self.linear_r(prompt_guids[-1].view(bsz, -1))
        cosine_similarities = torch.nn.functional.cosine_similarity(hybrid, last_resnet_layer, dim=-1)
        A = F.softmax(cosine_similarities, dim=0).unsqueeze(1)
        a = self.weights[0]
        b = self.weights[1]

      
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]
        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
        sum_prompt_guids = a*A*sum_prompt_guids + b*sum_prompt_guids
        sum_prompt_guids = sum_prompt_guids.view(bsz, -1)

        result = []
        for idx in range(12):  # 12
            prompt_gate = self.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)
            #prompt_gate = torch.ones_like(prompt_gate) / 4  
            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])
            # use gate mix aux image prompts
            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                ####
                #aux_prompt_gate = torch.ones_like(aux_prompt_gate) / 4
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result

    def softmax(self, x, dim):
        beta = nn.Parameter(torch.tensor(1.0))
        gamma = nn.Parameter(torch.tensor(1.0))
        max_val = torch.max(x, dim=dim, keepdim=True)[0] if dim is not None else torch.max(x, keepdim=True)[0]
        max_val = beta * max_val
        e_x = torch.exp(x - max_val)  # Subtracting the maximum value for numerical stability
        return e_x / e_x.sum(dim=dim, keepdim=True)*gamma






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

        if args.use_prompt:
            self.image_model = ImageModel()  # bsz, 6, 56, 56
            self.encoder_conv =  nn.Sequential(
                            nn.Linear(in_features=3840, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=4*2*768)
                            )
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])

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


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None,
                weight=None, input_ids_phrase=None, attention_mask_phrase=None):
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)


        feature_OriImg_FineGrained = self.get_resnet_feature(images)
        feature_OriImg_CoarseGrained = self.get_resnet_feature(aux_imgs.reshape(-1, 3, 224, 224))

        pic_ori = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori = torch.transpose(pic_ori, 1, 2)
        pic_ori = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori = self.linear_pic(pic_ori)
        pic_ori_ = torch.sum(pic_ori, dim=1)

        pic_ori_objects = torch.reshape(feature_OriImg_CoarseGrained, (-1, 2048, 49))
        pic_ori_objects = torch.transpose(pic_ori_objects, 1, 2)
        pic_ori_objects = torch.reshape(pic_ori_objects, (-1, 3, 49, 2048))
        pic_ori_objects = torch.sum(pic_ori_objects, dim=2)
        pic_ori_objects = self.linear_pic(pic_ori_objects)  # *weight_objects[:,:,0].reshape(-1,3,1)
        pic_ori_objects_ = torch.sum(pic_ori_objects, dim=1)  # .view(bsz, 16, 64)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                token_type_ids=token_type_ids,
                                past_key_values=prompt_guids,
                                return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        hidden_text = self.dropout(sequence_output)  # bsz, len, hidden


        output_phrases = self.bert(input_ids_phrase, attention_mask_phrase)
        hidden_phrases = output_phrases['last_hidden_state']

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_origin = self.linear_q_fine(pic_ori)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_ori_objects = self.linear_q_coarse(pic_ori_objects)
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)),
                                         dim=1)

        pic_ori_final = (pic_original+pic_ori_) * weight[:, 1].reshape(-1, 1) + \
                        (pic_original_objects+pic_ori_objects_) * weight[:, 0].reshape(-1,1)
        pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.args.max_seq, self.image2token_emb))
        emissions = self.fc(torch.relu(self.dropout(self.combine_linear(torch.cat([hidden_text, pic_ori], dim=-1)))))

        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='sum')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_resnet_feature(self, x):
        # 遍历resent的每一层的名字和层
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
        return x
    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, vae)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(sum=0.0, std=0.05)
    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]

        result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result


    def softmax(self, x, dim):
        beta = nn.Parameter(torch.tensor(1.0))
        gamma = nn.Parameter(torch.tensor(1.0))
        max_val = torch.max(x, dim=dim, keepdim=True)[0] if dim is not None else torch.max(x, keepdim=True)[0]
        max_val = beta * max_val
        e_x = torch.exp(x - max_val)  # Subtracting the maximum value for numerical stability
        return e_x / e_x.sum(dim=dim, keepdim=True)*gamma
