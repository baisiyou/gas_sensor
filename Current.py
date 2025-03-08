import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据加载
data = pd.read_csv("TiO2_acetone_temperature_current_1.5ppm_5V.csv")

# 2. 数据清洗：移除包含缺失值的行或使用合适的值填充
#   首先检查缺失值
print("原始数据缺失值数量:\n", data.isnull().sum())

# 移除包含缺失值的行
data = data.dropna()

# 再次检查缺失值
print("移除缺失值后的数据缺失值数量:\n", data.isnull().sum())

# 3. 数据类型转换：确保数值列是数值类型
#   'errors='coerce'' 会将无法转换为数值的值转换为 NaN
data['Temperature(°C)'] = pd.to_numeric(data['Temperature(°C)'], errors='coerce')
#data['Current(μA)'] = pd.to_numeric(data['Current(μA)'], errors='coerce') # Current(μA) 是目标变量，不需要在这里转换，后面会作为y值

# 4. 再次移除由于转换错误而产生的 NaN 行
data = data.dropna()

# 5. 特征选择
#   选择 'Temperature(°C)' 作为特征，预测 'Current(μA)'
X = data[['Temperature(°C)']]  #  只使用温度作为特征
y = data['Current(μA)']

# 6. 数据分割
#   将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 模型训练
#   使用 XGBoost 回归器进行训练
xgbr = xgb.XGBRegressor(objective='reg:squarederror',
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42)

xgbr.fit(X_train, y_train)

# 8. 模型评估
#   在测试集上评估模型性能
y_pred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("均方误差 (Mean Squared Error):", mse)
print("R 平方 (R-squared):", r2)

# 9. 预测 (示例)
#   使用训练好的模型进行预测
new_data = pd.DataFrame({'Temperature(°C)': [280]}) # 只需提供温度
predictions = xgbr.predict(new_data)
print("预测结果 (Predictions):", predictions)