import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
import math
from entmax import Sparsemax, SparsemaxBisect

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)

    
    def forward(self, x, aux_imgs=None):
        # full image prompt 全局图像特征
        prompt_guids= self.get_resnet_prompt(x)    # 4x[bsz, 256, 7, 7]

        # aux_imgs: bsz x 3(nums) x 3 x 224 x 224 对象特征 取3个对象？
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
        # 遍历resent的每一层的名字和层
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)    # (bsz, 256, 56, 56)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size() #256是channel，56是feature map的大小
                kernel = ft // 2 # 28
                # 28x28的feature map，每个feature map取一个点，然后做平均池化，得到一个值，然后把这个值作为prompt
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)    # (bsz, 256, 7, 7)
                prompt_guids.append(prompt_kv)   # conv2: (bsz, 256, 7, 7)

        return prompt_guids

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class HMNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HMNeTREModel, self).__init__()

        self.bert = BertModel.from_pretrained('../bert-base-uncased')
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.resnet = resnet50(pretrained=True)
        self.tokenizer = tokenizer

        # modify by zhang11.26
        # the attention mechanism for fine-grained features
        self.hidden_size = 768 * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_pic = nn.Linear(2048, self.hidden_size // 2)
        self.linear_final = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.linear_q_fine = nn.Linear(768, self.hidden_size // 2)
        self.linear_k_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_k_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)


        #3.17 try decouple the features
        combined_dim_low = self.hidden_size // 2
        self.len_l_p = 20
        self.len_v_o = 3

        #  Modality-specific encoder
        self.encoder_s_l = nn.Conv1d(self.hidden_size//2, self.hidden_size//2, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.hidden_size//2, self.hidden_size//2, kernel_size=1, padding=0, bias=False)
        # Modality-invariant encoder
        self.encoder_c = nn.Conv1d(self.hidden_size//2, self.hidden_size//2, kernel_size=1, padding=0, bias=False)
        self.decoder_l = nn.Conv1d(self.hidden_size, self.hidden_size//2, kernel_size=1, padding=0, bias=False)

        self.proj1_l_low_p = nn.Linear(combined_dim_low * self.len_l_p, combined_dim_low)
        self.proj2_l_low_p = nn.Linear(combined_dim_low, combined_dim_low * self.len_l_p)
        self.out_layer_l_low_p = nn.Linear(combined_dim_low * self.len_l_p, num_labels)
        self.proj1_v_low_o = nn.Linear(combined_dim_low * self.len_v_o, combined_dim_low)
        self.proj2_v_low_o = nn.Linear(combined_dim_low, combined_dim_low * self.len_v_o)
        self.out_layer_v_low_o = nn.Linear(combined_dim_low * self.len_v_o, num_labels)
        self.MSE = MSE()
        self.sparsemax = SparsemaxBisect(dim=-1)

        self.loss_fn = nn.CrossEntropyLoss()
        # if self.args.use_prompt:
        #     self.image_model = ImageModel()
        #     self.encoder_conv =  nn.Sequential(
        #                             nn.Linear(in_features=3840, out_features=800),
        #                             nn.Tanh(),
        #                             nn.Linear(in_features=800, out_features=4*2*768)
        #                         )
        #     self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])


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

        # if self.args.use_prompt:
        #     #modify by zhang11.7
        #     prompt_guids = self.get_visual_prompt(images, aux_imgs)#(12, 2, 16, 12, 4, 64)
        #     prompt_guids_length = prompt_guids[0][0].shape[2]
        #     prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
        #     prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        # else:
        #     prompt_guids = None
        #     prompt_attention_mask = attention_mask

        # modify by zhang11.26
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
                    output_attentions=True,
                    return_dict=True
        )
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden

        #3.17任务：这里做特征解耦,解耦出8个特征
        #模态相关特征
        s_l_s = self.encoder_s_l(last_hidden_state.permute(0, 2, 1))
        s_l_p = self.encoder_s_l(hidden_phrases.permute(0, 2, 1))
        s_v_g = self.encoder_s_v(pic_ori.permute(0, 2, 1))
        s_v_o = self.encoder_s_v(pic_ori_objects.permute(0, 2, 1))
        #共享编码器-->模态无关特征
        c_l_s = self.encoder_c(last_hidden_state.permute(0, 2, 1))
        c_l_p = self.encoder_c(hidden_phrases.permute(0, 2, 1))
        c_v_g = self.encoder_c(pic_ori.permute(0, 2, 1))
        c_v_o = self.encoder_c(pic_ori_objects.permute(0, 2, 1))
        c_list = [c_l_s, c_l_p, c_v_g, c_v_o]

        #重建loss
        recon_l_p = self.dropout(self.decoder_l(torch.cat([s_l_p, c_l_p], dim=1)))
        s_l_r_p = self.dropout(self.encoder_s_l(recon_l_p))
        loss_recon_l = self.MSE(recon_l_p.permute(0, 2, 1), hidden_phrases)
        #consistency loss
        loss_sl_slr = self.MSE(s_l_p.permute(0, 2, 1), s_l_r_p.permute(0, 2, 1))


        s_l_s = s_l_s.permute(0, 2, 1)
        s_l_p = s_l_p.permute(0, 2, 1)
        hidden_phrases = c_l_p.permute(0, 2, 1)

        hs_l_low_p = c_l_p.transpose(0, 1).contiguous().view(hidden_phrases.size(0), -1)
        repr_l_low_p = self.proj1_l_low_p(hs_l_low_p)
        hs_proj_l_low_p = self.proj2_l_low_p(
            F.dropout(F.relu(repr_l_low_p, inplace=True), p=0.5, training=self.training))
        hs_proj_l_low_p += hs_l_low_p
        logits_l_low_p = self.out_layer_l_low_p(hs_proj_l_low_p)
        loss_l_low_p = self.loss_fn(logits_l_low_p, labels.view(-1))

        hs_v_low_o = c_v_o.transpose(0, 1).contiguous().view(pic_ori_objects.size(0), -1)
        repr_v_low_o = self.proj1_v_low_o(hs_v_low_o)
        hs_proj_v_low_o = self.proj2_v_low_o(
            F.dropout(F.relu(repr_v_low_o, inplace=True), p=0.5, training=self.training))
        hs_proj_v_low_o += hs_v_low_o
        logits_v_low_o = self.out_layer_v_low_o(hs_proj_v_low_o)


        hidden_k_text = self.linear_k_fine(s_l_s)
        hidden_v_text = self.linear_v_fine(s_l_s)
        pic_q_origin = self.linear_q_fine(s_v_g.permute(0, 2, 1))
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_phrases = self.linear_k_coarse(s_l_p)
        hidden_v_phrases = self.linear_v_coarse(s_l_p)
        pic_q_ori_objects = self.linear_q_coarse(s_v_o.permute(0, 2, 1))
        # 4 与短语做注意力之后的对象特征
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)),
                                         dim=1)
        # 2号 短语特征
        hidden_phrases = torch.sum(hidden_phrases, dim=1)

        # 1号 实体对特征
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)



        x = torch.cat([hidden_phrases, entity_hidden_state,
                       pic_original+pic_original_objects], dim=-1)
        x = self.linear_final(self.dropout(x))
        x = self.dropout(x)

        logits = self.classifier(x)
        loss_task = self.loss_fn(logits, labels.view(-1))
        combined_loss = loss_task + loss_l_low_p + (loss_recon_l+loss_sl_slr)*0.1
        return combined_loss, repr_l_low_p, repr_v_low_o, logits_l_low_p, logits_v_low_o, logits, x



    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        #att_map = F.softmax(scores, dim=-1)
        att_map = self.sparsemax(scores)

        return torch.matmul(att_map, value)
    def get_resnet_feature(self, x):
        # 遍历resent的每一层的名字和层
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
        return x

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        # full image prompt
        # 获取全局图像和局部图像的resnet特征，每个特征有4块
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....
        # resize全局特征
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        # aux image prompts # 3 x (4 x [bsz, 256, 2, 2])
        #resize 局部特征
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]
        #再通过两层全连接层，再将全局特征和局部特征映射到4*2*768维？
        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        # 将特征映射到4*2*768维后，再将其分成4份，每份2*768维
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2

        result = []
        for idx in range(12):  # 12

            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)
            ####生成一个batchsize*4的张量，每个元素为0.25
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


class HMNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained('../bert-base-uncased')
        self.bert_config = self.bert.config
        self.resnet = resnet50(pretrained=True)

        # modify by zhang11.26
        # the attention mechanism for fine-grained features
        self.hidden_size = 768 * 2
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_pic = nn.Linear(2048, self.hidden_size // 2)
        self.linear_final = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.linear_q_fine = nn.Linear(768, self.hidden_size // 2)
        self.linear_k_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_k_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)


        #3.17 try decouple the features
        combined_dim_low = self.hidden_size // 2
        self.len_l_p = 20
        self.len_v_o = 3

        #  Modality-specific encoder
        self.encoder_s_l = nn.Conv1d(self.hidden_size//2, self.hidden_size//2, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(self.hidden_size//2, self.hidden_size//2, kernel_size=1, padding=0, bias=False)
        # Modality-invariant encoder
        self.encoder_c = nn.Conv1d(self.hidden_size//2, self.hidden_size//2, kernel_size=1, padding=0, bias=False)
        self.decoder_l = nn.Conv1d(self.hidden_size, self.hidden_size//2, kernel_size=1, padding=0, bias=False)

        self.proj1_l_low_p = nn.Linear(combined_dim_low * self.len_l_p, combined_dim_low)
        self.proj2_l_low_p = nn.Linear(combined_dim_low, combined_dim_low * self.len_l_p)
        self.out_layer_l_low_p = nn.Linear(combined_dim_low * self.len_l_p, num_labels)
        self.proj1_v_low_o = nn.Linear(combined_dim_low * self.len_v_o, combined_dim_low)
        self.proj2_v_low_o = nn.Linear(combined_dim_low, combined_dim_low * self.len_v_o)
        self.out_layer_v_low_o = nn.Linear(combined_dim_low * self.len_v_o, num_labels)
        self.MSE = MSE()
        self.sparsemax = SparsemaxBisect(dim=-1)

        self.loss_fn = nn.CrossEntropyLoss()


        self.num_labels = len(label_list)  # pad
        print(self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None,
                aux_imgs=None):
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                token_type_ids=token_type_ids,
                                past_key_values=prompt_guids,
                                return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)  # bsz, len, labels

        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
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
    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)  # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in
                            aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in
                            aux_prompt_guids]  # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768 * 2, dim=-1)  # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768 * 2, dim=-1) for aux_prompt_guid in
                                  aux_prompt_guids]  # 3x [4 x [bsz, 4, 768*2]]

        result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4  # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []  # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4  # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1),
                                                             split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1,
                                                                                              64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result











































class HMNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

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
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None):
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)    # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

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
