import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, args):
        super(LSTM_Model, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.input_dim)
        self.net = nn.LSTM(input_size=self.args.input_dim,
                           hidden_size=self.args.hidden_dim,
                           num_layers=self.args.num_layers,
                           batch_first=True,
                           dropout=self.args.dropout)
        self.logits = nn.Linear(self.args.hidden_dim, self.args.vocab_size)

    def forward(self, x):
        embedding_x = self.embedding(x)
        out, _ = self.net(embedding_x)
        return self.logits(out)

