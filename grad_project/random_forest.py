import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from grad_project.constants import OUTPUT_PATH, TEST_RATIO

raw_data = pd.read_csv('data/raw_data.csv', parse_dates=[0], index_col=0)
raw_data.dropna(inplace=True)
rmse = pd.DataFrame()
for column in raw_data.columns:
    print('processing {0}'.format(column))
    data = raw_data[[column]]
    print(data.head())
    column_rmse = pd.DataFrame()
    for index in range(1, 11):
        test_data = data.sample(frac=TEST_RATIO)
        train_data = data.drop(test_data.index)
        regressor = RandomForestRegressor(random_state=0)
        train_index_arr = np.array(train_data.index).reshape(-1, 1)
        regressor.fit(train_index_arr, train_data[column])
        test_index_arr = np.array(test_data.index).reshape(-1, 1)
        result = regressor.predict(test_index_arr)
        result_df = pd.DataFrame({column: pd.Series(result, index=test_data.index)})
        print('result')
        print(result_df.head())
        column_rmse[index] = ((result_df - test_data) ** 2).mean(0) ** 0.5
        print('column RMSE')
        print(column_rmse.head())
    rmse = rmse.append(column_rmse)
    print('current RMSE')
    print(rmse)
    rmse.to_csv('{0}random_forest_result.csv'.format(OUTPUT_PATH))
