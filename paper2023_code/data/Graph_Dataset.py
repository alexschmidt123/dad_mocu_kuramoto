import torch
import numpy as np
import json
import random
from torch_geometric.data import Data

with open('../Dataset/5o_type1.json', 'r') as fileObject:
    data2 = json.load(fileObject)

with open('../Dataset/5o_type2.json', 'r') as fileObject:
    data1 = json.load(fileObject)

data = data1 + data2
random.shuffle(data)

def getEdgeAtt(attr1, attr2, n):
    edge_attr = torch.zeros([2, n * (n - 1)])
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_attr[0, k] = attr1[i, j]
                edge_attr[1, k] = attr2[i, j]
                k = k + 1
    return edge_attr

data_list = []

for i in range(len(data)):
    x = np.asarray(data[i]['w'])
    x = torch.from_numpy(x.astype(np.float32))
    n = x.size()[0]
    x = x.unsqueeze(dim=1)

    edge_index = getEdgeAtt(np.tile(np.asarray([i for i in range(n)]), (n, 1)),
                            np.tile(np.asarray([[i] for i in range(n)]), (1, n)), n).long()
    edge_attr = getEdgeAtt(torch.from_numpy(np.asarray(data[i]['a_lower']).astype(np.float32)),
                           torch.from_numpy(np.asarray(data[i]['a_upper']).astype(np.float32)), n)

    y = torch.from_numpy(np.asarray(data[i]['mean_MOCU']).astype(np.float32))
    y = y.unsqueeze(dim=0).unsqueeze(dim=0)

    data_ = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
    data_list.append(data_)

train = data_list[0:70000]
test = data_list[70000:75000]

torch.save(train, '../Dataset/70000_5o_train.pth')
torch.save(test, '../Dataset/5000_5o_test.pth')