import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)


class MMREProcessor(object):
    def __init__(self, data_path, bert_name):
        self.data_path = data_path
        self.re_path = data_path['re_path']
        self.tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased', do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<s>', '</s>', '<o>', '</o>']})

    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """
        #读取文本数据
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h']) # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)
        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        #读取对象图片
        # aux image
        aux_path = self.data_path[mode+"_auximgs"]
        aux_imgs = torch.load(aux_path)

         # sample
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(words))), k=int(len(words)*sample_ratio))
            sample_words = [words[idx] for idx in sample_indexes]
            sample_relations = [relations[idx] for idx in sample_indexes]
            sample_heads = [heads[idx] for idx in sample_indexes]
            sample_tails = [tails[idx] for idx in sample_indexes]
            sample_imgids = [imgids[idx] for idx in sample_indexes]
            sample_dataid = [dataid[idx] for idx in sample_indexes]
            assert len(sample_words) == len(sample_relations) == len(sample_imgids), "{}, {}, {}".format(len(sample_words), len(sample_relations), len(sample_imgids))
            return {'words':sample_words, 'relations':sample_relations, 'heads':sample_heads, 'tails':sample_tails, \
                 'imgids':sample_imgids, 'dataid': sample_dataid, 'aux_imgs':aux_imgs}
        
        return {'words':words, 'relations':relations, 'heads':heads, 'tails':tails, 'imgids': imgids, 'dataid': dataid, 'aux_imgs':aux_imgs}
       

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict


class MMREDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, sample_ratio=1.0, mode="train") -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode]  if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode, sample_ratio)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer

        #modify the weight by11.26
        text_path = self.processor.data_path[mode]
        # self.weak_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_weak.txt'.format(mode))
        # self.strong_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_strong.txt'.format(mode))
        #load the correlation scores
        # with open(self.weak_ori, 'r', encoding='utf-8') as f_rel:
        #     lines = f_rel.readlines()
        #     self.weak_ori = {}
        #     for line in lines:
        #         img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
        #         self.weak_ori[img_id_key] = score
        # with open(self.strong_ori, 'r', encoding='utf-8') as f_rel:
        #     lines = f_rel.readlines()
        #     self.strong_ori = {}
        #     for line in lines:
        #         img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
        #         self.strong_ori[img_id_key] = score

        self.phrase_path = text_path.replace('ours_{}.txt'.format(mode), 'phrase_text_{}.json'.format(mode))
        f_grounding = open(self.phrase_path, 'r')
        self.phrase_data = json.load(f_grounding)

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], \
            self.data_dict['relations'][idx], self.data_dict['heads'][idx], \
            self.data_dict['tails'][idx], self.data_dict['imgids'][idx]

        item_id = self.data_dict['dataid'][idx]


        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict.data['input_ids'], \
            encode_dict.data['token_type_ids'], encode_dict.data['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), \
            torch.tensor(token_type_ids), torch.tensor(attention_mask)
        re_label = self.re_dict[relation]  # label to id

        #modify 11.26 获取短语级文本
        phrase= self.phrase_data[str(idx)]  # phrase
        encode_dict_ph = self.tokenizer(text=phrase, max_length=20, truncation=True, padding='max_length')
        input_ids_phrase, attention_mask_phrase = encode_dict_ph.data['input_ids'], encode_dict_ph.data['attention_mask']
        input_ids_phrase, attention_mask_phrase = torch.tensor(input_ids_phrase), torch.tensor(attention_mask_phrase)


         # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)#（3，224，224）

            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            if self.aux_img_path is not None:
                # process aux image
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths  = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                # discard more than 3 aux image
                # 选择3张aux image（这里的选择没有考虑相关性）
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)
                # padding zero if less than 3
                # 如果不足3张，用0矩阵填充
                for i in range(3-len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224)))
                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3



            return input_ids, token_type_ids, attention_mask,input_ids_phrase,\
                attention_mask_phrase, torch.tensor(re_label), image, aux_imgs


        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)






class MMPNERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased', do_lower_case=True)

    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
                                                                                    len(imgs))
        # load aux image
        aux_path = self.data_path[mode + "_auximgs"]
        aux_imgs = torch.load(aux_path)

        # load weak correlation between text and original image
        with open(self.data_path['%s_weight_weak' % mode], 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            weak_ori = {}
            for line in lines:
                img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
                weak_ori[img_id_key] = score

        # load strong correlation between text and original image
        with open(self.data_path['%s_weight_strong' % mode], 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            strong_ori = {}
            for line in lines:
                img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
                strong_ori[img_id_key] = score

        # load phrases for visual objects detection
        with open(self.data_path['%s_grounding_text' % mode], 'r', encoding='utf-8') as f_ner_phrase_text:
            data_phrase_text = json.load(f_ner_phrase_text)

        # sample data, only for low-resource
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words) * sample_ratio))
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(
                len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs, "aux_imgs": aux_imgs}

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs": aux_imgs,
                'weak_ori':weak_ori, 'strong_ori':strong_ori,'phrase_text':data_phrase_text}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]",
                      "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping

class MMPNERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, sample_ratio=1, mode='train', ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_img_path[mode]  if aux_img_path is not None else None
        self.mode = mode
        self.sample_ratio = sample_ratio
    
    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], self.data_dict['imgs'][idx]
        # get correlation coefficient
        weak_ori, strong_ori = self.data_dict['weak_ori'][img], self.data_dict['strong_ori'][img],
        phrase_text = self.data_dict['phrase_text'][img]
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx]*(self.max_seq-len(labels)-2)
        encode_dict_g = self.tokenizer.encode_plus(phrase_text, max_length=5, truncation=True, padding='max_length')
        input_ids_g, token_type_ids_g, attention_mask_g = encode_dict_g['input_ids'], encode_dict_g['token_type_ids'], \
        encode_dict_g['attention_mask']
        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)

            weight = [weak_ori, strong_ori]
            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                if img in self.data_dict['aux_imgs']:
                    aux_img_paths  = self.data_dict['aux_imgs'][img]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)

                for i in range(3-len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224))) 

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels), image, aux_imgs, \
                    torch.tensor(weight), torch.tensor(input_ids_g), torch.tensor(token_type_ids_g), torch.tensor(attention_mask_g)

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels)