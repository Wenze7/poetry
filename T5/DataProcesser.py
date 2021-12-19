import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import random


class DataProcesser(Dataset):
    def __init__(self, tokenizer, max_len, path):

        self.mode = 'CCPC' if 'CCPC' in path else 'other'
        print('Loading data from '+ path)
        # 选择训练集
        self.data = pd.read_csv(path, sep='\t')
        self.texts = self.data['content'].tolist()

        print('Data size :{}'.format(len(self.texts)))
        self.title = self.data['title'].tolist()
        self.keywords = self.data['keywords'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):

        text = str(self.texts[index])

        if self.mode == 'CCPC':
            keywords = str(self.keywords[index]).split(" ")
            keywords = keywords[random.randint(0, len(keywords) - 1):]
            random.shuffle(keywords)
            keywords = " ".join(keywords)
            #keywords = keywords.replace(" ", "[SPACE]")  # '屏开[SPACE]晴日[SPACE]春风[SPACE]绿苔'
            title = keywords
        else:
            title = str(self.texts[index])

        encoding_title = self.tokenizer.encode_plus(title,
                                                    truncation=True,
                                                    add_special_tokens=True,
                                                    max_length=10,
                                                    return_token_type_ids=True,
                                                    pad_to_max_length=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
        encoding_text = self.tokenizer(text,
                                       add_special_tokens=True,
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_token_type_ids=True,
                                       pad_to_max_length=True,
                                       return_attention_mask=True,
                                       return_tensors='pt')

        sample = {
            'texts': text,
            'keywords': title,
            'input_ids': encoding_title['input_ids'].flatten(),
            'attention_mask': encoding_title['attention_mask'].flatten(),
            'decoder_attention_mask': encoding_text['attention_mask'].flatten(),
            'label_ids': encoding_text['input_ids'].flatten()
        }

        # if self.mode != 'CCPC':
        #
        #     encoding_text = self.tokenizer.encode_plus(text,
        #                                           add_special_tokens=True,
        #                                           max_length=self.max_len,
        #                                           return_token_type_ids=True,
        #                                           pad_to_max_length=True,
        #                                           return_attention_mask=False,
        #                                           return_tensors='pt')
        #     sample = {
        #         'texts': text,
        #         'input_ids': encoding_text['input_ids'].flatten()
        #     }
        #
        # else:
        #
        #     keywords = str(self.keywords[index]).split(" ")
        #     keywords = keywords[random.randint(0, len(keywords) - 1):]
        #     random.shuffle(keywords)
        #     keywords = " ".join(keywords)
        #     keywords = keywords.replace(" ", "[SPACE]")  # '屏开[SPACE]晴日[SPACE]春风[SPACE]绿苔'
        #
        #
        # # print(len(sample['texts']))
        return sample

    def __len__(self):
        return len(self.texts)


def create_dataloader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

