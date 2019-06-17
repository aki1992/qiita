# -*- coding: utf-8 -*-
import numpy as np
import random
import torchvision
import pickle


if __name__ == '__main__':
    
    #Set a seed
    np.random.seed(0)
    random.seed(0)

    #Downlod dataset
    train_dataset = torchvision.datasets.MNIST(
        'MNIST',
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        'MNIST',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    
    reduction_indice = [0, 2, 4, 6, 8]
    train_data_list = []
    print('----Create Train Data List----')
    for label in np.arange(10):
        curr_data_list = [
            (feature, target)
            for feature, target in train_dataset
                if target == label
        ]
        original_size = len(curr_data_list)
        if label in reduction_indice:
            reduced_size = len(curr_data_list) // 100
            train_data_list += random.sample(curr_data_list, reduced_size)
            print(f'label:{label} #data(original):{original_size} #data(reduced):{reduced_size}')
        else:
            train_data_list += curr_data_list
            print(f'label:{label} #data(original):{original_size}')

    test_data_list = []
    print('----Create Test Data List----')
    for label in np.arange(10):
        curr_data_list = [
            (feature, target)
            for feature, target in test_dataset
                if target == label
        ]
        original_size = len(curr_data_list)
        test_data_list += curr_data_list
        print(f'label:{label} #data(original):{original_size}')

    #Save
    with open('train_data_list.pkl', mode='wb') as f:
        pickle.dump(train_data_list, f)

    with open('test_data_list.pkl', mode='wb') as f:
        pickle.dump(test_data_list, f)