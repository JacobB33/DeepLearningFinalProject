from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.data.DataLoader import get_train_dataset
import random
import torch
import numpy as np

set_backend("torch_cuda")

def get_train_test():
    dataset = get_train_dataset()
    train_len = int(len(dataset)*0.9)
    
    indicies = list(range(len(dataset)))
    random.shuffle(indicies)
    train_idx, test_idx = indicies[:train_len], indicies[train_len:]
    train_set, test_set = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in train_set:
        x_train.append(i[0])
        y_train.append(i[1])
    for i in test_set:
        x_test.append(i[0])
        y_test.append(i[1])
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    return x_train, x_test, y_train, y_test, test_idx

def main():
    x_train, x_test, y_train, y_test, test_idx = get_train_test()
    alpha = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]
    ridge = RidgeCV(alphas=alpha)
    preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
        )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )   
    pipeline.fit(x_train, y_train)
    scores = pipeline.predictt(x_test)
    print(f'mse = {np.mean((scores - y_test)**2)}')

if __name__ == '__main__':
    main()