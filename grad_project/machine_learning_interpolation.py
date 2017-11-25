import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from grad_project.constants import OUTPUT_PATH, TEST_RATIO

FEATURE_MAP = {
    'temp': ['date', 'rh', 'atm', 'rad', 'co2'],
    'rh': ['date', 'temp', 's_temp', 's_rh', 'rad'],
    's_temp': ['date', 'temp', 'rh', 's_rh', 'rad'],
    's_rh': ['date', 'temp', 'rh', 's_temp'],
    'ec': ['date', 's_temp', 's_rh', 'rad', 'co2'],
    'atm': ['date', 'temp', 'rh'],
    'rad': ['date', 'atm'],
    'co2': ['date', 'temp', 'rh', 'atm', 'rad'],
}

raw_data = pd.read_csv('data/raw_data.csv', parse_dates=[0])
raw_data.dropna(inplace=True)
date = raw_data['date'].apply(lambda x: int(x.to_datetime().timestamp()))
raw_data['date'] = date

for regressor_type in ['random_forest', 'mlp']:
    print('##### Start {0} #####'.format(regressor_type))
    rmse = pd.Series()
    r_square = pd.Series()
    for column in raw_data.columns[1:]:
        print('Start processing {0}'.format(column))
        columns = FEATURE_MAP.get(column).append(column)
        data = raw_data[FEATURE_MAP.get(column)]
        print(data.head())
        test_data = data.sample(frac=TEST_RATIO)
        train_data = data.drop(test_data.index)
        train_y = train_data[column]
        train_x = train_data.drop(column, axis=1)
        print('train_y')
        print(train_y.head())
        print('train_x')
        print(train_x.head())
        test_y = pd.DataFrame(test_data[column])
        test_x = test_data.drop(column, axis=1)
        if regressor_type == 'random_forest':
            regressor = RandomForestRegressor(random_state=0)
        else:
            regressor = MLPRegressor(
                learning_rate_init=0.001,
                max_iter=5000,
                batch_size=5000,
                early_stopping=True,
                hidden_layer_sizes=(256, 256),
                random_state=0, verbose=10
            )
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        print('normalized data')
        print('train_x')
        print(train_x)
        print('test_x')
        print(test_x)
        start = time.time()
        regressor.fit(train_x, train_y)
        training_time = time.time() - start
        raw_pred_y = regressor.predict(test_x)
        pred_y = pd.DataFrame({column: pd.Series(raw_pred_y, index=test_y.index)})
        print('##### Result #####')
        print('true Y')
        print(test_y.head())
        print('predicted Y')
        print(pred_y.head())
        rmse_value = (((test_y - pred_y) ** 2).mean(0) ** 0.5).values[0]
        rmse.set_value(column, rmse_value)
        rmse.set_value(column + '_time', training_time)
        r_square_value = regressor.score(test_x, test_y)
        r_square.set_value(column, r_square_value)
        print('RMSE')
        print(rmse.head(10))
        print('R-square')
        print(r_square.head(10))
        rmse.to_csv('{0}{1}_result.csv'.format(OUTPUT_PATH, regressor_type))
        r_square.to_csv('{0}{1}_result_r2.csv'.format(OUTPUT_PATH, regressor_type))
    print('##### {0} training Done #####'.format(regressor_type))
