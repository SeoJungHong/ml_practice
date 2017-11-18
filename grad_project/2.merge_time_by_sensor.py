import pandas as pd

from grad_project.constants import DATA_PATH, RAW_FILES

usecols_map = {
    1: [0, 1, 2, 6, 7, 9],
    2: [0, 1, 2, 4, 5, 7],
    3: [0, 1, 2, 6, 7, 9],
    4: [0, 1, 2, 4, 5, 7],
    5: [0, 1, 2, 6, 7, 9],
    6: [0, 1, 2, 6, 7, 9],
    7: [0, 1, 2, 6, 7, 9],
    8: [0, 1, 2, 4, 5, 7],
    9: [0, 1, 2, 6, 7, 9],
    10: [0, 1, 2, 4, 5, 6],
}

for i in range(1, 11):
    df_list = []
    print('merge data from sensor {0}'.format(i))
    for file in RAW_FILES:
        input_filename = '{0}{1}_{2}.csv'.format(DATA_PATH, file.split('.')[0], i)
        print('read {0}'.format(input_filename))
        df = pd.read_csv(input_filename, usecols=usecols_map.get(i))
        df_list.append(df)
    merged = pd.concat(df_list)
    if i < 10:  # 기상 센서가 아닐때
        prefix = 'N{0} '.format(i)
        merged.rename(columns={
            prefix + '날짜': '날짜',
            prefix + '온도': '온도',
            prefix + '습도': '습도',
            prefix + '대기압': '대기압',
            prefix + '조도': '조도',
            prefix + 'CO2': 'CO2',
        }, inplace=True)
        # column 순서 정렬
        columns = merged.columns.tolist()
        columns.remove('CO2')
        columns.append('CO2')
        merged = merged[columns]
    print('remove empty rows')
    merged["TMP"] = merged[merged.columns[0]]
    merged = merged[merged.TMP.notnull()]  # remove all NaT values
    merged.drop(["TMP"], axis=1, inplace=True)  # delete TMP again
    merged = merged[merged['날짜'] != '===============']
    merged[merged.columns[0]] = pd.DatetimeIndex(merged[merged.columns[0]])
    merged = merged[merged['날짜'] >= pd.datetime(2016, 8, 4, 12)]
    merged.sort_values(by='날짜')
    merged.info()
    output_filename = '{0}merged_{1}.csv'.format(DATA_PATH, i)
    merged.to_csv(output_filename, index=False)
    print('save merged data to {0}'.format(output_filename))
    print('--------------------')

print('Done')
