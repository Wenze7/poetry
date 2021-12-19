import pandas as pd

data = pd.read_csv('./dataset/' + 'CCPC' + '/' + 'CCPC' + '.csv', sep='\t')

train_num = int(len(data)*0.8)

train_CCPC = data[:train_num]
valid_CCPC = data[train_num:]

train_CCPC.to_csv('./dataset/' + 'CCPC' + '/' + 'CCPC_train' + '.csv', sep='\t')
valid_CCPC.to_csv('./dataset/' + 'CCPC' + '/' + 'CCPC_valid' + '.csv', sep='\t')
