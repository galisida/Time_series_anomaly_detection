import pymysql
import pandas as pd

file_path = 'dataset/all_data.csv'

data = pd.read_csv(file_path)
db = pymysql.connect(host='localhost', user='root', password='123123', db='yuheng')

query = """insert into catering_sale (num, date, sale) values (%s,%s,%s)"""


