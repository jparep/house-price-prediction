# Import Packages
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

###################
# CREATE FUNCTIONS
###################

# Load data from csv file function
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Separate Features vs Target variables function
def split_features_target(df, target_column='SalePrice'):
    features = df.drop(['Id', target_column], axis=1)
    target = df[target_column]
    return features, target

# Define Numerical and Categorical Features
def split_num_cat_features(features):
    num_features = features.select_dtypes(include=['int64', 'float64']).columns
    cat_features = features.select_dtypes(include=['onject']).columns
    return num_features, cat_features


    