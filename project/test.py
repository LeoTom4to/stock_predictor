import tushare as ts

# 设置Tushare Token
ts.set_token('2bf25750566da626ec1d3c56edec5c63546a685b4dfbed78e109917a')  # 使用你的Tushare token
pro = ts.pro_api()

# 获取股票数据
symbol = '601996.SH'
start_date = '20110101'
end_date = '20250229'
df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)

# 输出列名
print("列名:", df.columns)

# 输出数据框的前几行内容
print("数据内容:", df.head())
