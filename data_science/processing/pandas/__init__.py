import pandas as pd
import json

'''
learning material : https://www.runoob.com/pandas/pandas-tutorial.html
api : https://pandas.pydata.org/pandas-docs/stable/reference/index.html
'''


'data type (series)'
a = [1, 2, 3, 4, 5]
b = {'x': 'Google', 'y': 'Apple', 'z': 'Microsoft'}
print(pd.Series(data=a, name='a_series', dtype=int, index=range(5), copy=False))
print(pd.Series(data=b, name='b_series', dtype=str))

'data type (DataFrame)'
a = [['Google',10],['Runoob',12],['Wiki',13]]
print(pd.DataFrame(a, columns=['sites', 'age']))
b = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}
dfb = pd.DataFrame(b)
print(dfb)
print(dfb['Site']) # return column, pd.Series
print(dfb.loc[0]) # return row, pd.Series
print(dfb.loc[[0, 1]])
c = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
dfc = pd.DataFrame(c, index=['r1', 'r2'])
print(dfc)
print(dfc.loc['r1'])  # pd.Series
print(dfc.loc[['r1', 'r2']])

'csv'
csv_to_df = pd.read_csv('../../../asset/gre_word_list_1.csv')
print(csv_to_df)
print(csv_to_df.to_string())

nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]

df_to_csv = pd.DataFrame({'name': nme, 'site': st, 'age':ag})
df_to_csv.to_csv('../../asset/giant_info.csv')

print(csv_to_df.head()) # 5 rows as default
print(csv_to_df.head(1))
print(csv_to_df.tail())
print(csv_to_df.info()) # class, rangeindex, datacolumns, info per-column, dtype

'json'
df = pd.read_json('sites.json')

string_json = {
    "col1": {"row1": 1, "row2": 2, "row3": 3},
    "col2": {"row1": "x", "row2": "y", "row3": "z"}
}  # a dict&string form of json
df1 = pd.DataFrame(string_json)

url = 'https://static.runoob.com/download/sites.json' # a url form of json
df2 = pd.read_json(url)

# here is an example of nested json
df3 = pd.read_json('../../../asset/nested.json')  # not right
with open('../../../asset/nested.json', 'r') as f:
    data = json.loads(f.read()) # json.load loads file, while json.loads loads string
df_nested_list = pd.json_normalize(data, record_path=['student'])

# here is an example of nested mixed json (mixed : dict&array&list&etc.)
with open('../../../asset/nested_mix.json', 'r') as f:
    data = json.loads(f.read())
df_nested_mix_list = pd.json_normalize(data, record_path=['students'], meta=['class', ['info', 'president'], ['info', 'contacts', 'tel']])
print(df_nested_mix_list)

# when we just need to get an array of info in the nested json
import glom
df = pd.read_json('../../../asset/nested_mix.json')
print(df['student'].apply(lambda row : glom(row, 'grade.math')))


'data cleaning'
df = pd.read_csv('../../../asset/property-data.csv')
# how='all': drop when all is null; thresh: set how many should be not null to retain; subset: cols to check
new_df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print(df['NUM_BEDROOMS'].isnull)  # return pd.Series
print(df['NUM_BEDROOMS'].isnull)  # return pd.Series

# assign non-null types
missing_values = ['n/a', 'na', '-']
df = pd.read_csv('../../../asset/property-data.csv', na_values= missing_values)

# substitution
df.fillna(0, inplace=True)

# clean duplicated data
person = {
  "name": ['Google', 'Runoob', 'Runoob', 'Taobao'],
  "age": [50, 40, 40, 23]
}
df = pd.DataFrame(person)
print(df.duplicated())
df.drop_duplicates(inplace=True)


'data operation'
print(df['ST_NUM'].mean())
print(df['ST_NUM'].median())
print(df['ST_NUM'].mode())  # 众数
print(pd.to_datetime(df['Date']))  # adjust format
print(df['Date'].dt.week)
df.loc[2, 'SUT_NUM'] = 30  # the title line has no index, and the first line is of index 0
df.drop(2, inplace=True)   # delete row-2, after that, index will skip 2
print(df.groupby(df['ST_NUM'])['SUT_NAME'])
print(df['SUT_NAME'].str.split(','))

a = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
b = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
print(pd.merge(a, b, on='key'))  # join horizontally
print(pd.concat([a, b]))  # concatenation vertically





