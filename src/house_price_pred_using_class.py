import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
def load_data(file_path, delimiter=','):
    try:
        return pd.read_csv(file_path, delimiter=delimiter)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Define features
def define_features(X):
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    return num_features, cat_features

# Preprocessor
def preprocess(X, num_cat_features):
    num_features, cat_features = num_cat_features['num'], num_cat_features['cat']

    num_pipe = Pipeline([
        ('imputer', IterativeImputer(random_state=0)),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])

    return preprocessor.fit_transform(X)

# Model Selector and Tuning
class ModelSelector(BaseEstimator, TransformerMixin):
    def __init__(self, models, param_grids):
        self.models = models
        self.param_grids = param_grids
        self.best_model = None
        self.best_score = float('-inf')
        self.results = {}

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
        for name, model in self.models.items():
            grid_search = GridSearchCV(model, self.param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_

            y_pred = grid_search.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            self.results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'mse': mse,
                'r2': r2
            }

    def transform(self, X):
        return X

if __name__ == "__main__":
    file_path = "../data/raw/house_data.csv"  # Specify your data file path
    data = load_data(file_path)
    num_features, cat_features = define_features(data)
    preprocessed_data = preprocess(data, {'num': num_features, 'cat': cat_features})

    y = data['SalePrice']  # Specify your target column
    models = {
        'Random Forest': RandomForestRegressor(random_state=0),
        'Gradient Boosting': GradientBoostingRegressor(random_state=0)
        # Add other models here
    }
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.5]
        }
        # Add hyperparameters for other models here
    }

    model_selector = ModelSelector(models, param_grids)
    model_selector.fit(preprocessed_data, y)

    print("Model Evaluation Results:")
    for name, result in model_selector.results.items():
        print(f"{name}:")
        print(f"  Best Score (MSE): {result['best_score']:.2f}")
        print(f"  MSE on Validation Set: {result['mse']:.2f}")
        print(f"  R-squared on Validation Set: {result['r2']:.2f}")
        print(f"  Best Hyperparameters: {result['best_params']}")
        print()