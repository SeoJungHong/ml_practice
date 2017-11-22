import pandas as pd
from fbprophet import Prophet

from grad_project.constants import OUTPUT_PATH

raw_data = pd.read_csv('data/raw_data.csv', parse_dates=[0])
columns = raw_data.columns[1:]
date_column = raw_data.columns[0]
rmse = pd.DataFrame()
for column in columns:
    print('processing {0}'.format(column))
    data = raw_data[[date_column, column]]
    data.rename(columns={date_column: 'ds', column: 'y'}, inplace=True)
    print(data.head())
    column_rmse = pd.DataFrame()
    for index in range(1, 11):
        test_data = data.dropna().sample(frac=0.1)
        # train_data = data.drop(test_data.index)
        prophet = Prophet(yearly_seasonality=True, daily_seasonality=True)
        prophet.fit(data)
        print('prophet fitting done!')
        test_data.set_index('ds', inplace=True)
        test_period = pd.DataFrame({'ds': test_data.index})
        result = prophet.predict(test_period)
        print('raw_result')
        print(result.head())
        result_df = result[['ds', 'yhat']]
        result_df.rename(columns={'yhat': 'y'}, inplace=True)
        result_df.set_index('ds', inplace=True)
        print('result')
        print(result_df.head())
        print('test')
        print(test_data.head())
        column_rmse[index] = ((result_df - test_data) ** 2).mean(0) ** 0.5
        print('column RMSE')
        print(column_rmse.head())
    column_rmse.rename({'y': column}, inplace=True)
    rmse = rmse.append(column_rmse)
    print('current RMSE')
    print(rmse)
    rmse.to_csv('{0}prophet_result.csv'.format(OUTPUT_PATH))
