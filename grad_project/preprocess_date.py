import pandas as pd

from grad_project.constants import MERGED_CSV_FILES

# Round time in 10 min range and fill empty time rows
DATE_RANGE = pd.date_range('2016-08-04 12:00', '2017-08-31 23:50', freq='10min')
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
