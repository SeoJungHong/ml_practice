import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from grad_project.constants import OUTPUT_PATH, TEST_RATIO

raw_data = pd.read_csv('data/raw_data.csv', parse_dates=[0], index_col=0)
raw_data.dropna(inplace=True)

for regressor_type in ['random_forest', 'support_vector_machine']:
    print('***** Training Start *****')
    print(regressor_type)
    rmse = pd.DataFrame()
    r_square = pd.DataFrame()
    for column in raw_data.columns:
        print('processing {0}'.format(column))
        data = raw_data[[column]]
        print(data.head())
        column_rmse = pd.DataFrame()
        column_r_square = pd.DataFrame()
        for index in range(1, 11):
            print('run validation {0}'.format(index))
            test_data = data.sample(frac=TEST_RATIO)
            train_data = data.drop(test_data.index)
            if regressor_type == 'random_forest':
                regressor = RandomForestRegressor(random_state=0)
            else:
                regressor = SVR(kernel='rbf', C=1e3, gamma=0.1)
            train_index_arr = np.array(train_data.index).reshape(-1, 1)
            regressor.fit(train_index_arr, train_data[column])
            test_index_arr = np.array(test_data.index).reshape(-1, 1)
            result = regressor.predict(test_index_arr)
            result_df = pd.DataFrame({column: pd.Series(result, index=test_data.index)})
            column_rmse[index] = ((result_df - test_data) ** 2).mean(0) ** 0.5
            column_r_square[index] = r2_score(test_data, result_df, multioutput='raw_values')
        column_rmse['mean'] = column_rmse.mean(1)
        column_r_square['mean'] = column_r_square.mean(1)
        print('column RMSE')
        print(column_rmse.head())
        print('column R-square')
        print(column_r_square.head())
        rmse = rmse.append(column_rmse)
        r_square = r_square.append(column_r_square)
        rmse.to_csv('{0}{1}_result.csv'.format(OUTPUT_PATH, regressor_type))
        r_square.to_csv('{0}{1}_result_r2.csv'.format(OUTPUT_PATH, regressor_type))
