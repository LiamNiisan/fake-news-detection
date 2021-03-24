import os as os
import pandas as pd
import numpy as np
import fasttext
from tqdm import tqdm
from Model import Model


class FasttextModel(Model):

    def format(self):
        """
        This function adapts the format of data for fasttext
        """

        training_data_path = os.path.join(self.data_path, 'training')
        fasttext_data_path = os.path.join(self.data_path, 'fasttext')
        os.mkdir(fasttext_data_path)

        for i in tqdm(range(self.kfold_n)):

            t_dataset_path = os.path.join(
                training_data_path, 'dataset_' + str(i+1))
            f_dataset_path = os.path.join(
                fasttext_data_path, 'dataset_' + str(i+1))

            os.mkdir(f_dataset_path)

            t_data_path_train = os.path.join(t_dataset_path, 'train')
            f_data_path_train = os.path.join(f_dataset_path, 'data.train')

            self.xlsx_to_fasttext(t_data_path_train, f_data_path_train)

            t_data_path_test = os.path.join(t_dataset_path, 'test')
            f_data_path_test = os.path.join(f_dataset_path, 'data.valid')

            self.xlsx_to_fasttext(t_data_path_test, f_data_path_test)

    def xlsx_to_fasttext(self, xlsx_file_path, fasttext_file_path):
        """
        This function transforms the data into a fasttext file format

        Args:
            xlsx_file_path (string): location of excel files to transform
            fasttext_file_path (string): destination of transformated file
        """

        xlsx_file_true = pd.read_excel(
            os.path.join(xlsx_file_path, 'True.xlsx'))
        xlsx_file_fake = pd.read_excel(
            os.path.join(xlsx_file_path, 'Fake.xlsx'))

        f = open(fasttext_file_path, "w+", encoding="utf-8")

        for i in range(xlsx_file_true.shape[0]):
            row = '__label__true ' + xlsx_file_true[0].iloc[i] + '\n'
            f.write(row)

        for i in range(xlsx_file_fake.shape[0]):
            row = '__label__fake ' + xlsx_file_fake[0].iloc[i] + '\n'
            f.write(row)

        f.close()

    def train(self, epochs=100, learning_rate=0.5):
        """
        This function trains the fasttext model

        Args:
            epochs (int, optional): number of training epochs. Defaults to 100.
            learning_rate (float, optional): learning rate. Defaults to 0.5.
        """

        fasttext_data_path = os.path.join(self.data_path, 'fasttext')

        for i in tqdm(range(self.kfold_n)):

            f_dataset_path = os.path.join(
                fasttext_data_path, 'dataset_' + str(i+1), 'data.train')

            model = fasttext.train_supervised(
                input=f_dataset_path, epoch=epochs, lr=learning_rate)

            model.save_model(os.path.join(
                self.model_path, 'fasttext', 'model_' + str(i+1)))

    def test(self):
        """
        This function tests the models and generates an average f1 score
        """

        score = 0

        for i in tqdm(range(self.kfold_n)):

            f_model_path = os.path.join(
                self.model_path, 'fasttext', 'model_' + str(i+1))
            validation_dataset = os.path.join(
                self.data_path, 'fasttext', 'dataset_' + str(i+1), 'data.valid')

            model = fasttext.load_model(f_model_path)

            score += model.test(validation_dataset)[1]

        self.f1_score = score / self.kfold_n
