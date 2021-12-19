import math

import pandas as pd
import pickle
from tqdm import tqdm


def get_prob_dict(paths):
    path_name = ['CCPC', 'poetrySong', 'poetryTang']
    ppl_dict = {'CCPC': {}, 'poetrySong': {}, 'poetryTang': {}}
    char_dict = {'CCPC': {}, 'poetrySong': {}, 'poetryTang': {}}
    for i, path in enumerate(paths):
        data = pd.read_csv(path, sep='\t')
        contents = data['content'].tolist()
        for content in tqdm(contents):
            if type(content) is float:
                continue
            for idx in range(1, len(content)):
                if content[idx] == '。' or content[idx] == ',' or content[idx] == '，':continue
                if content[idx-1] == '。' or content[idx-1] == ',' or content[idx-1] == '，':continue
                now_bigram = content[idx-1: idx+1]
                if now_bigram not in ppl_dict[path_name[i]]:
                    ppl_dict[path_name[i]][now_bigram] = 0
                ppl_dict[path_name[i]][now_bigram] += 1
                now_char = content[idx-1]
                if now_char not in char_dict[path_name[i]]:
                    char_dict[path_name[i]][now_char] = 0
                char_dict[path_name[i]][now_char] +=1
        for k in ppl_dict[path_name[i]]:
            ppl_dict[path_name[i]][k] = ppl_dict[path_name[i]][k] / char_dict[path_name[i]][k[0]]

        print(len(ppl_dict[path_name[i]]))
    pickle.dump(ppl_dict, open('./dataset/ppl_dict/ppl_dict.pkl', 'wb'))


def cal_ppl(path, mode):
    ppl_dict = pickle.load(open('./dataset/ppl_dict/ppl_dict.pkl', 'rb'))
    mode_dict = ppl_dict[mode]
    data = pd.read_csv(path, sep='\t')
    print(data)
    contents = data['0'].tolist()
    ppl = 0
    for content in contents:
        now_ppl = 1
        now_bigram_cnt = 0
        for idx in range(1, len(content)):
            if content[idx] == '。' or content[idx] == ',':continue
            if content[idx-1] == '。' or content[idx-1] == ',':continue
            now_bigram = content[idx - 1: idx + 1]
            if now_bigram not in mode_dict:
                now_ppl *= 1/0.001
            else:
                now_ppl *= 1/mode_dict[now_bigram]
            now_bigram_cnt += 1
        now_ppl = math.pow(now_ppl, 1/now_bigram_cnt)
        ppl += now_ppl
    print('ppl:{}'.format(ppl/len(contents)))

if __name__ == '__main__':
    # path_CCPC = './dataset/CCPC/CCPC.csv'
    # path_poetrySong = './dataset/poetrySong/poetrySong.csv'
    # path_poetryTang = './dataset/poetryTang/poetryTang.csv'
    # paths = [path_CCPC, path_poetrySong, path_poetryTang]
    # get_prob_dict(paths)
    cal_ppl('./dataset/inference_res/res_poetrySong.csv', 'poetrySong')