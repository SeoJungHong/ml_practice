import pandas as pd

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'
RAW_FILES = [
    '1610.xlsx',
    '1611.xlsx',
    '1612.xlsx',
    '1701.xlsx',
    '1702.xlsx',
    '1703.xlsx',
    '1704.xlsx',
    '1705.xlsx',
    '1706.xlsx',
    '1707.xlsx',
    '1708.xlsx'
]
DATE_RANGE = pd.date_range('2016-10-02 00:10', '2017-08-31 23:50', freq='10min')
TEST_RATIO = 0.3
