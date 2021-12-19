import torch
import torch.nn.functional as F
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from DataProcesser import *
from nltk.translate.bleu_score import sentence_bleu


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=True, type=bool, help='是否使用GPU')
    parser.add_argument('--mode', default="CCPC", type=str, help='训练模式(唐诗:tang、宋诗：song、宋词：ci、关键词诗：CCPC')
    parser.add_argument('--batch_size', default=64, type=int, help='生成古诗的个数')
    parser.add_argument('--seq_num', default=3, type=int)
    parser.add_argument('--title_max_len', default=10, type=int)
    parser.add_argument('--beam_size', default=3, type=int)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--generate_max_len', default=64, type=int, help='生成古诗的最大长度')
    parser.add_argument('--repetition_penalty', default=2.0, type=float, help='重复处罚率')
    parser.add_argument('--keyword_penalty', default=1.2, type=float, help='关键词处罚率')
    parser.add_argument('--top_k', default=5, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--valid', type=bool, default=True, help='测试还是预测')

    return parser.parse_args()
from tqdm import tqdm

def valid(args, model, data_loader):
    all_bleu = 0
    for step, sample in enumerate(data_loader):
        title = sample['input_ids'].to(args.device, dtype=torch.long)
        attention_mask = sample['attention_mask'].to(args.device, dtype=torch.long)
        sample_outputs = model.generate(
            input_ids=title,
            attention_mask=attention_mask,
            max_length=args.generate_max_len,
            num_beams=args.beam_size,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            length_penalty=1.0,
            do_sample=True,
            early_stopping=True,
            num_return_sequences=1
        )

        pre_texts = []
        now_batch_bleu = 0
        for sample in sample_outputs:
            print(tokenizer.decode(sample, skip_special_tokens=True))
            pre_texts.append(tokenizer.decode(sample, skip_special_tokens=True))

    return pre_texts

import pandas as pd

if __name__ == '__main__':

    args = get_args()
    args.device = "cuda:1" if torch.cuda.is_available() and args.gpu else "cpu"
    tokenizer = T5Tokenizer.from_pretrained('./ModelConfig/T5_chinese')
    model = T5ForConditionalGeneration.from_pretrained("./ModelConfig/model_" + args.mode + "/final_model").to(args.device)

    model.eval()
    if args.valid == True:
        valid_path = './dataset/CCPC_test.csv'
        valid_set = DataProcesser(tokenizer, args.max_len, valid_path)
        data_loader = create_dataloader(valid_set, args.batch_size)
        pre_texts = valid(args, model, data_loader)
        res = pd.DataFrame()
        res[0] = pre_texts
        res.to_csv('./dataset/inference_res/res_' + args.mode + '.csv', header=None, index=None)
    else:
        print('开始生成古诗，输入CTRL + Z退出')
        while True:
            content = input("输入的字符为:")
            encoding_title = tokenizer.encode_plus(content,
                                                   truncation=True,
                                                   add_special_tokens=True,
                                                   max_length=10,
                                                   return_token_type_ids=True,
                                                   pad_to_max_length=True,
                                                   return_attention_mask=True,
                                                   return_tensors='pt')
            title = encoding_title['input_ids'].to(args.device, dtype=torch.long)
            attention_mask = encoding_title['attention_mask'].to(args.device, dtype=torch.long)
            sample_outputs = model.generate(
                input_ids=title,
                attention_mask=attention_mask,
                max_length=args.generate_max_len,
                num_beams=args.beam_size,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                length_penalty=1,
                do_sample=True,
                early_stopping=True,
                num_return_sequences=3
            )

            poems = []

            for sample in sample_outputs:
                poems.append(tokenizer.decode(sample, skip_special_tokens=True))

            for i, poem in enumerate(poems):
                if args.mode != "CCPC":
                    print("生成的第{}个诗为：{}".format(i + 1, content + poem))
                else:
                    print("生成的第{}个诗为：{}".format(i + 1, poem))

