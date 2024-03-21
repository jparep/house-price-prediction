# Load the dataset from csv file
df = pd.read_csv("../data/raw/house_data.csv")
# Separate features and target variable
X = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']
print(X.columns.tolist())
# Define the columns after removing 'Id' and target variable
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns
# Initialize transformer for numerical variabels
num_trans = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=373)
# Initialize transfomer for categorical variables
cat_trans = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=373, initial_strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Combine the transformers into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', num_trans, num_cols),
    ('cat', cat_trans, cat_cols)
])
