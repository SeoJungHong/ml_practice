from functools import reduce

import pandas as pd

DATA_PATH = 'data/'

file_header_map = {
    '1607.xlsx': 5,
    '1608.xlsx': 5,
    '1609.xlsx': 5,
    '1610.xlsx': 6,
    '1611.xlsx': 6,
    '1612.xlsx': 6,
}
sensor_column_nums = [0, 10, 8, 10, 8, 10, 10, 10, 8, 10, 8]
for file in file_header_map.keys():
    print('Reading ' + file)
    for i in range(1, 11):
        if i < 10:
            print('parse data from sensor {0}'.format(i))
            col_start_index = reduce((lambda x, y: x + y), sensor_column_nums[:i]) + (i - 1)
            col_end_index = col_start_index + sensor_column_nums[i]
            column_index_list = list(range(col_start_index, col_end_index))
        else:
            print('parse data from weather sensor')
            col_start_index = reduce((lambda x, y: x + y), sensor_column_nums[:i]) + (i - 1) + 11
            col_end_index = col_start_index + sensor_column_nums[i]
            column_index_list = list(range(col_start_index, col_end_index))
        print('read column index {0} ({1})'.format(column_index_list, len(column_index_list)))
        data = pd.read_excel(
            DATA_PATH + file,
            header=file_header_map.get(file),
            usecols=column_index_list,
            encoding='utf-8'
        )
        print(data.head())
        print(data.columns.tolist())
        output_filename = '{0}{1}_{2}.csv'.format(DATA_PATH, file.split('.')[0], i)
        data.to_csv(output_filename, index=False)
        print('save data to {0}'.format(output_filename))
        print('--------------------')

for i in range(1, 11):
    df_list = []
    print('merge data from sensor {0}'.format(i))
    for file in file_header_map.keys():
        input_filename = '{0}{1}_{2}.csv'.format(DATA_PATH, file.split('.')[0], i)
        print('read {0}'.format(input_filename))
        df = pd.read_csv(input_filename)
        df.info()
        df_list.append(df)
    print('* Merge and sort values')
    merged = pd.concat(df_list)
    merged.sort_values(by=merged.columns[0], inplace=True)
    merged.info()
    output_filename = '{0}merged_{1}.csv'.format(DATA_PATH, i)
    merged.to_csv(output_filename, index=False)
    print('save merged data to {0}'.format(output_filename))
    print('--------------------')

print('Done')
