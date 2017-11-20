import pandas as pd

from grad_project.constants import DATA_PATH, DATE_RANGE

raw_data_file = '{0}raw_data.csv'.format(DATA_PATH)


def linear(data):
    linear_df = data.interpolate()
    # linear_df.to_csv('{0}processed_linear.csv'.format(DATA_PATH))
    return linear_df


def spline(data):
    spline_df = data.interpolate('spline', order=3, s=0)
    # spline_df.to_csv('{0}processed_spline.csv'.format(DATA_PATH))
    return spline_df


if __name__ == "__main__":
    df = pd.read_csv(raw_data_file, index_col=0, parse_dates=[0])
    print('raw data')
    df.info()
    # 10-fold cross validation
    for interpolation_method in [linear, spline]:
        rmse = pd.DataFrame()
        print('start new method: ' + interpolation_method.__name__)
        for index in range(1, 11):
            test_data = df.dropna().sample(frac=0.1)
            train_data = df.drop(test_data.index)
            train_data = train_data.loc[DATE_RANGE]
            interpolated = interpolation_method(train_data)
            rmse[index] = ((interpolated - test_data) ** 2).mean(0) ** 0.5
        print(rmse)
