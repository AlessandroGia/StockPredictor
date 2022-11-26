import sklearn.model_selection as sk
import pandas as pd
import numpy as np
import os


class LoadCsv:
    def __init__(self):
        self.__csvs_path: [str] = []
        for file in os.listdir('dataset'):
            self.__csvs_path.append(os.path.join('dataset', file))

    @staticmethod
    def __clean_features(df: pd.DataFrame) -> None:
        df.pop('Date')
        df.pop('Day')
        df.pop('Month')
        df.pop('OutcomeDopoSettimana')

    @staticmethod
    def __get_target(df: pd.DataFrame) -> pd.Series:
        return df.pop('OutcomeGiornoDopo')

    def __extract_features(self) -> [np.ndarray, np.ndarray]:
        df = pd.read_csv(self.__csvs_path[0])
        self.__clean_features(df)
        targets = self.__get_target(df)
        return np.array(df), np.array(targets)

    def load_data(self) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return sk.train_test_split(*self.__extract_features())
