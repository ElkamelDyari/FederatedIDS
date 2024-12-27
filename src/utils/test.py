from src.utils import FeatureEngineering, DataPreprocessing
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DIR_PATH = os.path.join('..', 'data', 'raw', 'MachineLearningCVE.csv')
DIR_PATH = r"C:\Users\Dyari Elkamel\Desktop\FL_IDS_MLOPS\data\raw\MachineLearningCVE.csv"
data = pd.read_csv(DIR_PATH)
print(data.shape)
data = DataPreprocessing.preprocessing(data)
print(data.shape)
print(data.columns)
print(data.head())
data = FeatureEngineering.feature_engineering(data)
print(data.shape)