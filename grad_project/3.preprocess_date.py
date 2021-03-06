import pandas as pd

from grad_project.constants import DATA_PATH, DATE_RANGE

# Round time in 10 min range and fill empty time rows
MERGED_CSV_FILES = ['{0}merged_{1}.csv'.format(DATA_PATH, i) for i in range(1, 11)]

for filename in MERGED_CSV_FILES:
    print('reading {0}'.format(filename))
    df = pd.read_csv(filename, index_col=0)
    print('before')
    print(df.head())
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.round(freq='10min')
    df = df.loc[DATE_RANGE]
    df = df[~df.index.duplicated()]
    print('after')
    print(df.head())
    df.info()
    df.to_csv(filename)
