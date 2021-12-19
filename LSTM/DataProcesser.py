import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import random
import numpy as np


class DataProcesser(Dataset):
    def __init__(self, tokenizer, max_len, path):
        self.mode = 'CCPC' if 'CCPC' in path else 'other'
        print('Loading data from '+path)
        # 选择训练集
        self.data = pd.read_csv(path, sep='\t')
        self.texts = self.data['content'].tolist()

        print('Data size :{}'.format(len(self.texts)))

        self.keywords = self.data['keywords'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):

        text = str(self.texts[index])

        if self.mode != 'CCPC':
            encoding_text = self.tokenizer.encode_plus(text,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  return_token_type_ids=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=False,
                                                  return_tensors='pt')
            sample = {
                'texts': text,
                'input_ids': encoding_text['input_ids'].flatten()
            }

        else:

            keywords = str(self.keywords[index]).split(" ")
            keywords = keywords[random.randint(0, len(keywords) - 1):]
            random.shuffle(keywords)
            keywords = " ".join(keywords)
            keywords = keywords.replace(" ", "[SPACE]")  # '屏开[SPACE]晴日[SPACE]春风[SPACE]绿苔'

            encoding_text = self.tokenizer.encode_plus(keywords + " [SEP] " + text,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  return_token_type_ids=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=False,
                                                  return_tensors='pt')

            keyword_tokens = self.tokenizer.tokenize(keywords)[:18]
            keyword_tokens = ['[CLS]'] + keyword_tokens + ['[SEP]']
            keyword_tokens = max(0, 20-len(keyword_tokens)) * ['[PAD]'] + keyword_tokens
            keyword_tokens = self.tokenizer.convert_tokens_to_ids(keyword_tokens)

            keyword_mask = np.ones(len(keyword_tokens))

            keyword_mask[np.array(keyword_tokens) == 0] = 0

            keyword_tokens = torch.tensor(keyword_tokens)
            keyword_mask = torch.tensor(keyword_mask)



            key_len = len(self.tokenizer.tokenize(keywords + " [SEP] ")) + 1  # CLS
            cnt_len = len(self.tokenizer.tokenize(text)) + 1  # SEP

            encoding_token = self.tokenizer.encode_plus("[KEYWORD]" * key_len + " [CONTENT] " * cnt_len,
                                                        add_special_tokens=False,
                                                        max_length=self.max_len,
                                                        return_token_type_ids=False,
                                                        pad_to_max_length=True,
                                                        return_attention_mask=False,
                                                        return_tensors='pt')

            sample = {
                'texts': text,
                'keywords': keywords,
                'input_ids': encoding_text['input_ids'].flatten(),
                'tokens': encoding_token['input_ids'].flatten(),
                'title': keyword_tokens.flatten(),
                'title_mask': keyword_mask.flatten(),
            }

        return sample

    def __len__(self):
        return len(self.texts)


def create_dataloader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

