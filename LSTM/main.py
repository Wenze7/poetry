import torch
import torch.nn as nn
import argparse
from transformers import BertTokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from DataProcesser import *
from LSTM_Model import *
from Train import *
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='songci', help='选择数据集')
    parser.add_argument('--max_len', type=int, default=50, help='诗词的最大长度')
    parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--batch_size', type=int, default=64, help='mini batch')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='L2 norm')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--warm_up_ratio', type=float, default=0.0, help='warm up')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.device = "cuda:1" if torch.cuda.is_available() and args.gpu else "cpu"
    tokenizer = BertTokenizer(vocab_file="../ModelConfig/model_" + args.mode + "/vocab.txt", do_lower_case=False)
    tokenizer.add_tokens("[SPACE]", special_tokens=True)
    tokenizer.add_tokens("[KEYWORD]", special_tokens=True)
    tokenizer.add_tokens("[CONTENT]", special_tokens=True)
    model_config = GPT2Config.from_json_file("../ModelConfig/model_" + args.mode + "/config.json")
    args.vocab_size = len(tokenizer)
    train_path = '../dataset/' + args.mode + '/' + args.mode + '.csv'
    trainset = DataProcesser(tokenizer, model_config.n_ctx, train_path)
    train_loader = create_dataloader(trainset, args.batch_size)

    args.content_id = tokenizer.convert_tokens_to_ids("[CONTENT]")


    model = LSTM_Model(args)
    # model = nn.DataParallel(model)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warm_up_ratio * total_steps,
        num_training_steps=total_steps
    )

    Criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
    train(args, model, optimizer, scheduler, Criterion, train_loader)
    # output_dir = "./ModelConfig/model_" + args.mode + "/"
    # # print('training finished')
    # # if not os.path.exists(output_dir + 'final_model'):
    # #     os.mkdir(output_dir + 'final_model')
    # # torch.save(model.state_dict(), output_dir + 'final_model/model')
    # model.load_state_dict(torch.load(output_dir + 'final_model/model'))
