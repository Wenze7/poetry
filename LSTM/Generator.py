import torch
import torch.nn.functional as F
import argparse
from transformers import BertTokenizer, GPT2Config
from LSTM_Model import *
from nltk.translate.bleu_score import sentence_bleu
import copy
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=True, type=bool, help='是否使用GPU')
    parser.add_argument('--mode', default="CCPC", type=str, help='训练模式(唐诗:tang、宋诗：song、宋词：ci、关键词诗：CCPC')
    parser.add_argument('--batch_size', default=3, type=int, help='生成古诗的个数')
    parser.add_argument('--seq_num', default=3, type=int)
    parser.add_argument('--title_max_len', default=10, type=int)
    parser.add_argument('--beam_size', default=3, type=int)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--generate_max_len', default=64, type=int, help='生成古诗的最大长度')
    parser.add_argument('--repetition_penalty', default=1.4, type=float, help='重复处罚率')
    parser.add_argument('--keyword_penalty', default=0.8, type=float, help='关键词处罚率')
    parser.add_argument('--top_k', default=5, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--valid', type=bool, default=True, help='测试还是预测')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    return parser.parse_args()


def top_k_top_p(logits, top_k, top_p, filter_value=-float("Inf")):

    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))
    if top_k > 0:
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][-1, None]
            logit[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False

        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def predict_one_sample(args, model, tokenizer, keyword):
    keyword_tokens = tokenizer.tokenize(keyword.replace(" ", "[SPACE]"))

    # 给生成的诗词留下空
    if len(keyword_tokens) > args.max_len - 3 - args.generate_max_len:
        keyword_tokens = keyword_tokens[:args.max_len - 3 - args.generate_max_len]

    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")

    token_type_tensors = None

    if args.mode == "CCPC":
        content_id = tokenizer.convert_tokens_to_ids("[CONTENT]")
        keyword_id = tokenizer.convert_tokens_to_ids("[KEYWORD]")
        keyword_tokens = ["[CLS]"] + keyword_tokens + ["[SEP]"]
        token_type_ids = [[keyword_id] * len(keyword_tokens) for _ in range(args.beam_size)]
        token_type_tensors = torch.tensor(token_type_ids).long().to(args.device)
        next_token_type = torch.tensor([[content_id] for _ in range(args.beam_size)]).long().to(args.device)
    else:
        keyword_tokens = ["[CLS]"] + keyword_tokens

    input_ids = tokenizer.convert_tokens_to_ids(keyword_tokens)
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.beam_size)]

    input_tensors = torch.tensor(input_ids).long().to(args.device)

    generated = []
    keywords = []

    for keyword in input_ids[0]:
        keywords.append([keyword]*args.beam_size)

    finish_set = set()


    for step in range(args.generate_max_len):

        outputs = model(input_tensors)
        next_token_logits = outputs[:, -1, :]


        # 重复词语惩罚
        for index in range(args.beam_size):
            for token_id in set([token_ids[index] for token_ids in generated]):
                next_token_logits[index][token_id] /= args.repetition_penalty

        # 关键字惩罚
        for index in range(args.beam_size):
            for token_id in set([token_ids[index] for token_ids in keywords]):
                next_token_logits[index][token_id] /= args.keyword_penalty

        # unk设置为-inf
        for next_token_logit in next_token_logits:
            next_token_logit[unk_id] = -float("Inf")

        filter_logits = top_k_top_p(next_token_logits, top_k=args.top_k, top_p=args.top_p)
        next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)

        for index, token_id in enumerate(next_tokens[:, 0]):
            if token_id == sep_id:
                finish_set.add(index)
        finish_flag = True

        for index in range(args.beam_size):
            if index not in finish_set:
                finish_flag = False
                break
        if finish_flag:
            break
        generated.append([token.item() for token in next_tokens[:, 0]])

        input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
        if args.mode == "CCPC":
            token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)

    candidate_responses = []
    for index in range(args.beam_size):
        responses = []
        for token_index in range(len(generated)):
            if generated[token_index][index] != sep_id:
                responses.append(generated[token_index][index])
            else:
                break
        candidate_responses.append(
            "".join(tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[SPACE]", ""))
    return candidate_responses

if __name__ == '__main__':


    args = get_args()
    args.device = "cuda:1" if torch.cuda.is_available() and args.gpu else "cpu"
    tokenizer = BertTokenizer(vocab_file="./ModelConfig/model_" + args.mode + "/vocab.txt", do_lower_case=False)

    tokenizer.add_tokens("[SPACE]", special_tokens=True)
    tokenizer.add_tokens("[KEYWORD]", special_tokens=True)
    tokenizer.add_tokens("[CONTENT]", special_tokens=True)
    args.vocab_size = len(tokenizer)
    model_config = GPT2Config.from_json_file("../ModelConfig/model_" + args.mode + "/config.json")
    model = LSTM_Model(args)
    model.load_state_dict(torch.load("./ModelConfig/model_" + args.mode + "/model_epoch10/model"))
    args.max_len = model_config.n_ctx
    model.to(args.device)
    model.eval()

    if args.valid == True:
        valid_path = './dataset/CCPC_test.csv'
        data = pd.read_csv(valid_path, sep='\t')
        keywords = data['keywords'].tolist()
        contents = data['content'].tolist()
        res = pd.DataFrame()
        res_poems = []
        for i in range(len(data)):
            if args.mode == 'CCPC':
                content = keywords[i]
            else:
                content = contents[i][:3]

            args.generate_max_len = min(args.generate_max_len, args.max_len - 5 - len(content))
            poem = predict_one_sample(args, model, tokenizer, content)[0]
            if args.mode != "CCPC":
                res_poems.append(content + poem)
            else:
                res_poems.append(poem)

        res[0] = res_poems
        res.to_csv('./dataset/inference_res/res_10epoch_' + args.mode + '.csv', header=None, index=None)
    else:

        print('开始生成古诗，输入CTRL + Z退出')
        while True:
            content = input('请输入关键字：')
            args.generate_max_len = min(args.generate_max_len, args.max_len - 5 - len(content))
            poems = predict_one_sample(args, model, tokenizer, content)
            for i, poem in enumerate(poems):
                if args.mode != "CCPC":
                    print("生成的第{}个诗为：{}".format(i + 1, content + poem))
                else:
                    print("生成的第{}个诗为：{}".format(i + 1, poem))

