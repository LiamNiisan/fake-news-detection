import os as os
from tqdm import tqdm
import pandas as pd
import numpy as np
from Preprocessing import Preprocessing
import decomposition


class Model:

    def __init__(self, path, kfold_n):

        self.data_path = os.path.join(path, 'data')
        self.model_path = os.path.join(path, 'models')
        self.kfold_n = kfold_n
        self.f1_score = 0

    def preprocess(self):
        """
        This function preprocesses the data for training
        """        
        
        prp = Preprocessing()

        path_true = os.path.join(self.data_path, 'True.xlsx')
        path_fake = os.path.join(self.data_path, 'Fake.xlsx')

        corpus_true = prp.get_data(path_true)
        corpus_fake = prp.get_data(path_fake)

        decomposition.main(corpus_true, corpus_fake,
                           self.kfold_n, self.data_path)

    def format(self, kfold_n):
        pass

    def train(self):
        pass

    def test(self):
        pass
