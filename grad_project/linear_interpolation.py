import pandas as pd

from grad_project.constants import DATA_PATH, DATE_RANGE, OUTPUT_PATH, TEST_RATIO

raw_data_file = '{0}raw_data.csv'.format(DATA_PATH)


def linear(data):
    # Time base data 이므로 method = time 입력
    result_df = data.interpolate(method='time')
    result_df.to_csv('{0}processed_linear.csv'.format(DATA_PATH))
    return result_df


def spline(data):
    result_df = data.interpolate('spline', order=3, s=0)
    result_df.to_csv('{0}processed_spline.csv'.format(DATA_PATH))
    return result_df


def polynomial(data):
    result_df = data.interpolate('polynomial', order=3, s=0)
    result_df.to_csv('{0}processed_polynomial.csv'.format(DATA_PATH))
    return result_df


if __name__ == "__main__":
    df = pd.read_csv(raw_data_file, index_col=0, parse_dates=[0])
    print('raw data')
    df.info()
    for interpolation_method in [linear, spline, polynomial]:
        rmse = pd.DataFrame()
        print('start new method: ' + interpolation_method.__name__)
        for index in range(1, 11):
            test_data = df.dropna().sample(frac=TEST_RATIO)
            train_data = df.drop(test_data.index)
            train_data = train_data.loc[DATE_RANGE]
            interpolated = interpolation_method(train_data)
            rmse[index] = ((interpolated - test_data) ** 2).mean(0) ** 0.5
        print(rmse)
        rmse.to_csv('{0}{1}_result.csv'.format(OUTPUT_PATH, interpolation_method.__name__))
