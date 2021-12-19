import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import random
import numpy as np


class DataProcesser(Dataset):
    def __init__(self, tokenizer, max_len, data_num, zh_data_path, en_data_path):

        print('Loading data from '+zh_data_path+' and '+en_data_path)
        if data_num > 0:
            self.zh_texts = self.read_text(zh_data_path)[:data_num]
            self.en_texts = self.read_text(en_data_path)[:data_num]
        else:
            self.zh_texts = self.read_text(zh_data_path)
            self.en_texts = self.read_text(en_data_path)
        print('Data size :{}'.format(len(self.zh_texts)))

        self.tokenizer = tokenizer
        self.max_len = max_len

    def read_text(self, path):
        data_text = []
        fo = open(path, 'r')
        for row in fo:
            data_text.append(row.strip('\n'))
        fo.close()
        return data_text

    def __getitem__(self, index):
        zh_text = str(self.zh_texts[index])
        en_text = str(self.en_texts[index])
        encoding_zh_text = self.tokenizer.encode_plus(zh_text,
                                                      add_special_tokens=True,
                                                      max_length=self.max_len,
                                                      return_token_type_ids=True,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')
        encoding_en_text = self.tokenizer.encode_plus(en_text,
                                                      add_special_tokens=True,
                                                      max_length=self.max_len,
                                                      return_token_type_ids=True,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')

        sample = {
            'zh_text': zh_text,
            'en_text': en_text,
            'zh_input_ids': encoding_zh_text['input_ids'].flatten(),
            'en_input_ids': encoding_en_text['input_ids'].flatten(),
            'zh_attention_mask': encoding_zh_text['attention_mask'].flatten(),
            'en_attention_mask': encoding_en_text['attention_mask'].flatten()
        }

        return sample

    def __len__(self):
        return len(self.zh_texts)


def create_dataloader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader