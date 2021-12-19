# 诗词生成

- DataProcesser.py 数据处理生成dataloader
- Model.py 模型
- ppl.py 困惑度计算
- Train.py 训练

上传到github之前对文件目录格式进行了调整，但是内部代码逻辑是正确的。

## GPT2

重新训练了一个gpt2，效果最好

## T5

使用MengziT5，由于诗词的特殊性，诗名和诗的内容可能并不诗太相关，导致cross-attention时效果太差。另外，通过T5进行了机器翻译的训练。

## LSTM

使用了一层的LSTM，作为baseline

## 参考内容

- [liucongg/GPT2-NewsTitle: Chinese NewsTitle Generation Project by GPT2.带有超级详细注释的中文GPT2新闻标题生成项目。 (github.com)](https://github.com/liucongg/GPT2-NewsTitle)
- [Langboat/Mengzi: Mengzi Pretrained Models (github.com)](https://github.com/Langboat/Mengzi)
- [Datasets/CCPC at master · THUNLP-AIPoet/Datasets (github.com)](https://github.com/THUNLP-AIPoet/Datasets/tree/master/CCPC)