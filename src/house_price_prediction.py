import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Separate numerical and categorical features
def get_num_cat_features(features):
    num_features = features.select_dtypes(include=['number']).columns
    cat_features = features.select_dtypes(include=['object']).columns
    return num_features, cat_features

# Create a preprocessor pipeline
def create_preprocessor(num_features, cat_features):
    num_pipe = Pipeline([
        ('imputer', IterativeImputer(random_state=0)),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])

# Tune hyperparameters using GridSearchCV
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Split data into train and test sets
def split_train_test_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=0)

# Model selection function
def model_selection(models, param_grids, features, target):
    results = {}
    num_features, cat_features = get_num_cat_features(features)
    preprocessor = create_preprocessor(num_features=num_features,
                                       cat_features=cat_features)
    
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        best_model = tune_hyperparameters(pipe, param_grids[name], features, target)
        
        X_train, X_test, y_train, y_test = split_train_test_data(features, target)
        best_model.fit(X_train, y_train)
        
        y_pred = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'model': best_model, 'mse': mse, 'r2': r2}
        
    return results

if __name__ == "__main__":
    file_path = "../data/raw/house_data.csv"  # Replace with your actual data file
    df = load_data(file_path)
    
    # Assuming you have a 'target' column in your data
    target = df['SalePrice']
    features = df.drop(columns=['Id', 'SalePrice'], axis=1)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=0),
        'Gradient Boosting': GradientBoostingRegressor(random_state=0)
    }
    
    param_grids = {
        'Linear Regression': {},
        'Random Forest': {'model__n_estimators': [100, 200, 300],
                          'model__max_depth': [None, 10, 20],
                          'model__min_samples_split': [2, 5, 10]},
        'Gradient Boosting': {'model__n_estimators': [100, 200, 300],
                               'model__learning_rate': [0.05, 0.1, 0.2]}
    }
    
    results = model_selection(models, param_grids, features, target)
    
    for name, result in results.items():
        print(f"{name} - Mean Squared Error: {result['mse']}, R-squared: {result['r2']}")
