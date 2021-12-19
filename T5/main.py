import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from DataProcesser2 import *
from Train import *
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='trans', help='选择数据集')
    parser.add_argument('--max_len', type=int, default=64, help='诗词的最大长度')
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='mini batch')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='L2 norm')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--warm_up_ratio', type=float, default=0.0, help='warm up')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--gpu_num', type=str, default='0')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    args.device = "cuda:1" if torch.cuda.is_available() and args.gpu else "cpu"
    model = T5ForConditionalGeneration.from_pretrained('./ModelConfig/T5_chinese').to(args.device)
    tokenizer = T5Tokenizer.from_pretrained('./ModelConfig/T5_chinese')

    # tokenizer.add_tokens(['[SPACE]'])
    # model.resize_token_embeddings(len(tokenizer))
    # tokenizer.add_special_tokens({'additional_special_tokens': ["[SPACE]"]})
    if args.mode == 'trans':
        zh_path = './dataset/trans/news-commentary-v13.zh-en.zh'
        en_path = './dataset/trans/news-commentary-v13.zh-en.en'
        trainset = DataProcesser(tokenizer, args.max_len, -1, zh_path, en_path)
        data_loader = create_dataloader(trainset, args.batch_size)
    else:
        train_path = './dataset/' + args.mode + '/' + args.mode + '_train.csv'

        trainset = DataProcesser(tokenizer, args.max_len, train_path)
        data_loader = create_dataloader(trainset, args.batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_steps = len(data_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warm_up_ratio * total_steps,
        num_training_steps=total_steps
    )

    train(args, model, optimizer, scheduler, data_loader, tokenizer)



