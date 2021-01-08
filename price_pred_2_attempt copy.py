import pandas as pd
import xgboost
import numpy as np
from sklearn.impute import SimpleImputer


def one_hot(columns, main):
    data = pd.DataFrame
    for i, col in enumerate(columns):
        print(main[col])
        dummy_df = pd.get_dummies(main[col], drop_first=True)
        print(dummy_df)
        breakpoint()
        main = main.drop([col], axis=1)
        if i == 0:
            data = dummy_df
        else:
            data = pd.concat([data, dummy_df], axis=1)
    data = pd.concat([main, data], axis=1)
    return data


data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv').drop(['Id'], axis=1)
Id = list(data_test.Id)     # save Id column for later use
data_test.drop(['Id'], axis=1)          # well, dropping Id column
main_data = pd.concat([data_train, data_test], axis=0)      # create a big data set with both training and test data

categorical = list(main_data.select_dtypes(include='object'))       # select only categorical values (non-numerical)
numerical = list(main_data.select_dtypes(exclude='object'))         # the opposite

for name, ser in zip(main_data.columns, main_data.isnull().sum()):   # name-column names; ser-series with missing values
    if name == 'SalePrice':
        continue
    if ser:
        if ser/14.6 > 50:
            main_data = main_data.drop([name], axis=1)
            try:
                categorical.remove(name)
            except ValueError:
                numerical.remove(name)

        elif name in categorical:
            main_data[name] = main_data[name].fillna(main_data[name].mode()[0])

        elif name in numerical:
            main_data[name] = main_data[name].fillna(main_data[name].mean())


final_data = one_hot(categorical, main_data)            # apply one hot encoding
final_data = final_data.loc[:, ~final_data.columns.duplicated()]        # remove duplicate rows
train_df = final_data.iloc[:1460]       # 0-1460 -- train data, else - test
test_df = final_data.iloc[1460:]
# select data:
train_X = train_df.drop(['SalePrice'], axis=1)
train_Y = train_df.SalePrice
val_X = test_df.drop(['SalePrice'], axis=1)
sub_sample = pd.read_csv('sample_submission.csv')

model = xgboost.XGBRegressor(n_estimators=1000,
                             learning_rate=0.05,
                             n_jobs=4)
model.fit(train_X, train_Y,
          verbose=False)
preds = model.predict(val_X)

submission = pd.DataFrame({'Id': Id,
                          'SalePrice': preds})
submission.to_csv('submission.csv', index=False)




