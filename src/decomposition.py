#----------------- Libraries -------------------#
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from Preprocessing import Preprocessing


def kfold_decompose(data, kfold_n):
    """
    This function uses kfold to split the data.

    Args:
        data (list): The data to split
        kfold_n (int): number of fragments to be split

    Returns:
        list[dict]: a list of the split datasets
    """

    X = np.array(data)
    kf = KFold(n_splits=kfold_n, random_state=2, shuffle=True)
    data_output = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        data_output.append({'train': X_train, 'test': X_test})

    return data_output


def create_folders(data_label_true, data_label_fake, kfold_n, data_path):
    """
    This function fragments the data and creates the repositories to store it.

    Args:
        data_label_true (list[dict]): true data text
        data_label_fake (list[dict]): fake data text
        kfold_n (int): number of data splits with kfold
    """
    # cwd = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(cwd, os.pardir, 'data')
    training = os.path.join(data_path, 'training')
    os.mkdir(training)
    prp = Preprocessing()

    for i in tqdm(range(kfold_n)):

        dataset_path = os.path.join(training, 'dataset_' + str(i+1))

        os.mkdir(dataset_path)
        os.mkdir(os.path.join(dataset_path, 'test'))
        os.mkdir(os.path.join(dataset_path, 'train'))
        os.mkdir(os.path.join(dataset_path, 'train', 'vc'))

        pd.DataFrame(data_label_true[i]['test']).to_excel(
            os.path.join(dataset_path, 'test', 'True.xlsx'), index=False)
        pd.DataFrame(data_label_fake[i]['test']).to_excel(
            os.path.join(dataset_path, 'test', 'Fake.xlsx'), index=False)

        X_train, X_val = train_test_split(
            data_label_true[i]['train'], test_size=0.20, random_state=1)
        pd.DataFrame(prp.preprocess(X_train)).to_excel(
            os.path.join(dataset_path, 'train', 'True.xlsx'), index=False)
        pd.DataFrame(X_val).to_excel(os.path.join(
            dataset_path, 'train', 'vc', 'True.xlsx'), index=False)

        X_train, X_val = train_test_split(
            data_label_fake[i]['train'], test_size=0.20, random_state=1)
        pd.DataFrame(prp.preprocess(X_train)).to_excel(
            os.path.join(dataset_path, 'train', 'Fake.xlsx'), index=False)
        pd.DataFrame(X_val).to_excel(os.path.join(
            dataset_path, 'train', 'vc', 'Fake.xlsx'), index=False)


def main(true_set, fake_set, kfold_n, data_path):
    """
    This function takes the text dataset of true and fake news and splits it with kfolds 
    and creates the repositories for it.

    Args:
        true_set (numpy): list of text of true label dataset
        fake_set (numpy): list of text of fake label dataset
        kfold_n (int, optional): kfold stplit. Defaults to 5.
    """

    data_label_true = kfold_decompose(true_set, kfold_n)
    data_label_fake = kfold_decompose(fake_set, kfold_n)

    create_folders(data_label_true, data_label_fake, kfold_n, data_path)
