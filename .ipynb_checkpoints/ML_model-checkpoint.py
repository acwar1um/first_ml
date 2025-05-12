import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Загрузка данных
data = pd.read_csv("data.csv")

# Кодирование категориальных признаков
data_encoded = data.copy()
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])

# Матрица корреляций
corr_matrix = data_encoded.corr().round(2)

# Вывод значимых корреляций (порог > 0.3)
significant_corrs = corr_matrix[
    (corr_matrix.abs() > 0.3) & (corr_matrix != 1.0)
].dropna(how='all').dropna(how='all', axis=1)

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("Значимые корреляции:")
print(significant_corrs)
significant_corrs.to_csv("full_corr_matrix.csv")