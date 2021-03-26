import os as os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tqdm import tqdm
from Model import Model


class TensorflowModel(Model):

    def __init__(self, path, kfold_n):

        super().__init__(path, kfold_n)
        self.vectorize_layer = None

    def format(self):
        """
        This function adapts the format of data for tensorflow
        """

        training_data_path = os.path.join(self.data_path, 'training')
        tensorflow_data_path = os.path.join(self.data_path, 'tensorflow')
        os.mkdir(tensorflow_data_path)

        for i in tqdm(range(self.kfold_n)):

            t_dataset_path = os.path.join(
                training_data_path, 'dataset_' + str(i+1))
            tf_dataset_path = os.path.join(
                tensorflow_data_path, 'dataset_' + str(i+1))

            os.mkdir(tf_dataset_path)

            t_data_path_train = os.path.join(t_dataset_path, 'train')
            tf_data_path_train = os.path.join(tf_dataset_path, 'train')
            os.mkdir(tf_data_path_train)

            self.xlsx_to_tensorflow(t_data_path_train, tf_data_path_train)

            t_data_path_valid = os.path.join(t_dataset_path, 'train', 'vc')
            tf_data_path_valid = os.path.join(tf_dataset_path, 'valid')
            os.mkdir(tf_data_path_valid)

            self.xlsx_to_tensorflow(t_data_path_valid, tf_data_path_valid)

            t_data_path_test = os.path.join(t_dataset_path, 'test')
            tf_data_path_test = os.path.join(tf_dataset_path, 'test')
            os.mkdir(tf_data_path_test)

            self.xlsx_to_tensorflow(t_data_path_test, tf_data_path_test)

    def xlsx_to_tensorflow(self, xlsx_file_path, tensorflow_repository_path):
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

        true_data_path = os.path.join(tensorflow_repository_path, 'true')
        os.mkdir(true_data_path)

        for i in range(xlsx_file_true.shape[0]):
            file_path = os.path.join(true_data_path, str(i) + '.txt')
            f = open(file_path, "w+", encoding="utf-8")
            f.write(xlsx_file_true[0].iloc[i])
            f.close()

        fake_data_path = os.path.join(tensorflow_repository_path, 'fake')
        os.mkdir(fake_data_path)

        for i in range(xlsx_file_fake.shape[0]):
            file_path = os.path.join(fake_data_path, str(i) + '.txt')
            f = open(file_path, "w+", encoding="utf-8")
            f.write(xlsx_file_fake[0].iloc[i])
            f.close()

    def train(self, epochs=40, max_features = 10000, sequence_length = 250, embedding_dim = 16, seed = 42, batch_size = 32):
        """
        This function trains the tensorflow model

        Args:
            epochs (int, optional): number of training epochs. Defaults to 100.
            learning_rate (float, optional): learning rate. Defaults to 0.5.
        """

        tensorflow_data_path = os.path.join(self.data_path, 'tensorflow')

        for i in tqdm(range(self.kfold_n)):

            tf_dataset_path = os.path.join(
                tensorflow_data_path, 'dataset_' + str(i+1))


            raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
                os.path.join(tf_dataset_path, 'train'),
                batch_size=batch_size)

            raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
                os.path.join(tf_dataset_path, 'valid'),
                batch_size=batch_size)

            self.vectorize_layer = TextVectorization(
                max_tokens=max_features,
                output_mode='int',
                output_sequence_length=sequence_length)

            # Make a text-only dataset (without labels), then call adapt
            train_text = raw_train_ds.map(lambda x, y: x)
            self.vectorize_layer.adapt(train_text)

            train_ds = raw_train_ds.map(self.vectorize_text)
            val_ds = raw_val_ds.map(self.vectorize_text)

            AUTOTUNE = tf.data.AUTOTUNE

            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            model = tf.keras.Sequential([
                layers.Embedding(max_features + 1, embedding_dim),
                layers.Dropout(0.2),
                layers.GlobalAveragePooling1D(),
                layers.Dropout(0.2),
                layers.Dense(1)])

            model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                          optimizer='adam',
                          metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs)

            export_model = tf.keras.Sequential([
                self.vectorize_layer,
                model,
                layers.Activation('sigmoid')
            ])

            export_model.compile(
                loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy', tf.keras.metrics.Precision()]
            )

            loss, accuracy, precision = export_model.evaluate(raw_val_ds)

            export_model.save(os.path.join(
                self.model_path, 'tensorflow', 'model_' + str(i+1)))


    def vectorize_text(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label

    def test(self):
        """
        This function tests the models and generates an average f1 score
        """

        total_accuracy = 0
        total_precision = 0
        tensorflow_data_path = os.path.join(self.data_path, 'tensorflow')

        for i in tqdm(range(self.kfold_n)):

            tf_model_path = os.path.join(
                self.model_path, 'tensorflow', 'model_' + str(i+1))
            tf_dataset_path = os.path.join(
                tensorflow_data_path, 'dataset_' + str(i+1))

            model = tf.keras.models.load_model(tf_model_path)

            batch_size = 32

            raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
                os.path.join(tf_dataset_path, 'test'),
                batch_size=batch_size)

            loss, accuracy, precision = model.evaluate(raw_test_ds)
            total_accuracy += accuracy
            total_precision += precision

        self.accuracy = total_accuracy / self.kfold_n
        self.precision = total_precision / self.kfold_n
