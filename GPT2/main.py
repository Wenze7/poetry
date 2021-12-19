import argparse
from transformers import BertTokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from DataProcesser import *
from Model import *
from Train import *
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='songci', help='选择数据集')
    parser.add_argument('--max_len', type=int, default=50, help='诗词的最大长度')
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='L2 norm')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--warm_up_ratio', type=float, default=0.0, help='warm up')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"

    tokenizer = BertTokenizer(vocab_file="./ModelConfig/model_" + args.mode + "/vocab.txt", do_lower_case=False)
    tokenizer.add_tokens("[SPACE]", special_tokens=True)
    tokenizer.add_tokens("[KEYWORD]", special_tokens=True)
    tokenizer.add_tokens("[CONTENT]", special_tokens=True)
    model_config = GPT2Config.from_json_file("./ModelConfig/model_" + args.mode + "/config.json")
    train_path = './dataset/' + args.mode + '/' + args.mode + '_train.csv'
    trainset = DataProcesser(tokenizer, model_config.n_ctx, train_path)
    train_loader = create_dataloader(trainset, args.batch_size)
    model = MYGPT2LMHeadModel(model_config)
    model.to(args.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warm_up_ratio * total_steps,
        num_training_steps=total_steps
    )
    content_id = tokenizer.convert_tokens_to_ids("[CONTENT]")
    train(args, model, optimizer, scheduler, train_loader, content_id)













