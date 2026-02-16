import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


file_path = "/Users/alexey/programming/Big-Data-Analysis-Labs/attestation/DS_attestation_middle_variants/10-Tаблица 1.csv"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values=["null"]
)

for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".")
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Первые 20 строк:")
print(df.head(20))

print("\nИнформация о датасете:")
print(df.info())

print("\nСтатистика по числовым признакам:")
print(df.describe())

print("\nКатегориальные признаки:")
categorical_cols = df.select_dtypes(include=["object"]).columns

if len(categorical_cols) > 0:
    print(df[categorical_cols].describe())
else:
    print("Категориальных признаков нет.")


print("\nКоличество наблюдений:", df.shape[0])
print("Количество признаков:", df.shape[1])

print("\nПропуски:")
print(df.isnull().sum())

target = "medv"

plt.figure()
df[target].hist(bins=20)
plt.title("Распределение целевой переменной")
plt.xlabel("medv")
plt.ylabel("Частота")
plt.show()

plt.figure()
sns.boxplot(x=df[target])
plt.title("Boxplot целевой переменной")
plt.show()

plt.figure()
plt.scatter(df["nox"], df[target])
plt.xlabel("nox")
plt.ylabel("medv")
plt.title("nox vs medv")
plt.show()

# ЧАСТЬ 2. ПОДГОТОВКА ДАННЫХ

df = df.fillna(df.median())

categorical_cols = ["chas", "rad"]
numeric_cols = [col for col in df.columns if col not in categorical_cols + [target]]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nСтолбцы после One-Hot Encoding:")
print(df_encoded.columns)

print("\nФорма датасета после кодирования:", df_encoded.shape)

correlation = df_encoded.corr()

target_corr = correlation[target].sort_values(ascending=False)

print("\nТОП-10 корреляций с целевой:")
print(target_corr.head(100))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, cmap="coolwarm")
plt.title("Heatmap корреляционной матрицы")
plt.show()

# ЧАСТЬ 3. МОДЕЛЬ

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=6
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)
best_model = None
best_r2_score = -999

for k in range(1, 15):     
    print(f"-----k={k}------")
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nМетрики:")
    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Train R2:", train_r2)
    print("Test R2:", test_r2)

    if test_r2 > best_r2_score:
        best_r2_score = test_r2
        best_model = model



with open("knn_model_variant6.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nМодель сохранена в knn_model_variant6.pkl")
