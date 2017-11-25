import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from grad_project.constants import OUTPUT_PATH, TEST_RATIO

FEATURE_MAP = {
    'temp': ['date', 'rh', 'atm', 'rad', 'co2'],
    'rh': ['date', 'temp', 's_temp', 's_rh', 'rad'],
    's_temp': ['date', 'temp', 'rh', 's_rh', 'rad'],
    's_rh': ['date', 'temp', 'rh', 's_temp'],
    'ec': ['date', 's_temp', 's_rh', 'rad', 'co2'],
    'atm': ['date', 'temp', 'rh'],
    'rad': ['date'],
    'co2': ['date', 'temp', 'rh', 'atm', 'rad'],
}

raw_data = pd.read_csv('data/raw_data.csv', parse_dates=[0])
raw_data.dropna(inplace=True)
date = raw_data['date'].apply(lambda x: int(x.to_datetime().timestamp()))
raw_data['date'] = date
rmse = pd.DataFrame()
r_square = pd.DataFrame()
for column in raw_data.columns[1:]:
    print('Start processing {0}'.format(column))
    columns = FEATURE_MAP.get(column).append(column)
    data = raw_data[FEATURE_MAP.get(column)]
    print(data.head())
    column_rmse = pd.DataFrame()
    column_r_square = pd.DataFrame()
    for index in range(1, 11):
        print('run validation {0}'.format(index))
        test_data = data.sample(frac=TEST_RATIO)
        train_data = data.drop(test_data.index)
        regressor = RandomForestRegressor(random_state=0)
        train_y = train_data[column]
        train_x = train_data.drop(column, axis=1)
        print('label')
        print(train_y.head())
        print('data')
        print(train_x.head())
        regressor.fit(train_x, train_y)
        test_y = pd.DataFrame(test_data[column])
        test_x = test_data.drop(column, axis=1)
        raw_pred_y = regressor.predict(test_x)
        pred_y = pd.DataFrame({column: pd.Series(raw_pred_y, index=test_y.index)})
        print('##### Result #####')
        print('true Y')
        print(test_y.head())
        print('predicted Y')
        print(pred_y.head())
        column_rmse[index] = ((test_y - pred_y) ** 2).mean(0) ** 0.5
        column_r_square[index] = r2_score(test_y, pred_y, multioutput='raw_values')
    column_rmse['mean'] = column_rmse.mean(1)
    column_r_square['mean'] = column_r_square.mean(1)
    print('column RMSE')
    print(column_rmse.head())
    print('column R-square')
    print(column_r_square.head())
    rmse = rmse.append(column_rmse)
    r_square = r_square.append(column_r_square)
    rmse.to_csv('{0}random_forest_result.csv'.format(OUTPUT_PATH))
    r_square.to_csv('{0}random_forest_result_r2.csv'.format(OUTPUT_PATH))
