import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


all_test = pd.read_csv('test.csv')  #not used
all_train = pd.read_csv('train.csv')
X = all_train.drop('SalePrice', axis=1)
Y = all_train.SalePrice

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=65)

train_X_plus = train_X.copy()
val_X_plus = val_X.copy()

missing_cols = [col for col in all_train.columns if all_train[col].isnull().any()]
for col in missing_cols:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    val_X_plus[col + '_was_missing'] = val_X_plus[col].isnull()

categorical_cols = list(train_X_plus.select_dtypes(include='object'))   # cat col names
numerical_cols = list(train_X_plus.select_dtypes(exclude='object'))     # num col names

num_transformer = SimpleImputer(strategy='median')      # just imputation
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),       # first imputation
    ('onehot', OneHotEncoder(handle_unknown='ignore'))         # then one hot
                                 ])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

model = RandomForestRegressor(n_estimators=100, random_state=65)

pipe = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', model)
])

pipe.fit(train_X_plus, train_Y)
preds = pd.DataFrame(pipe.predict(val_X_plus))
scores = -1 * cross_val_score(pipe, train_X_plus, train_Y, cv=5, scoring='neg_mean_absolute_error')
print(mean_absolute_error(val_Y, preds))
# print(val_Y.head(), '\n------------------------------')
# print(preds.head())
print(scores.mean())


