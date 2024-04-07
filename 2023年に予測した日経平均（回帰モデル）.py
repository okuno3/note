# 2023年：回帰モデルは未来の予測に対して十分に精度が出るとは限りません。
# 2024年：実際に運用に足る予測モデルとするにはこの式を作成しただけでは不十分でした。

import pandas as pd

import datetime as dt
import pandas as pd
import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# データを取得
df = pdr.DataReader("^NKX", "stooq")

# 日付でソート
df.sort_index(inplace=True)

# 日付を数値に変換
df['Date'] = df.index.map(dt.datetime.toordinal)

# 使用する特徴量と目的変数を選択
features = ['Date']
target = 'Close'

# 特徴量と目的変数を取得
X = df[features].values.reshape(-1, 1)
y = df[target]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 線形回帰モデルを作成
model = LinearRegression()
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# 未来の日付を生成
future_dates = pd.date_range(start=dt.datetime(2024, 1, 13), end=dt.datetime(2024, 12, 31))

# 未来の日付を数値に変換
future_dates_numeric = future_dates.map(dt.datetime.toordinal)

# 未来の日付に対する予測
future_predictions = model.predict(future_dates_numeric.values.reshape(-1, 1))

# グラフで表示
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='実際のデータ')
plt.scatter(future_dates, future_predictions, color='red', label='未来の予測')
plt.title('Nikkei 225 Index - 未来の予測 (線形回帰モデル)')
plt.xlabel('日付')
plt.ylabel('終値')
plt.legend()
plt.show()