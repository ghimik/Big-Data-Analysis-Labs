import matplotlib.pyplot as plt

automl_models = [
    {"name": "Extra Trees", "MSE": 2.364376e7, "R2": 0.9579},
    {"name": "XGBoost", "MSE": 2.574019e7, "R2": 0.9548},
    {"name": "Decision Tree", "MSE": 2.776610e7, "R2": 0.9509},
    {"name": "Random Forest", "MSE": 3.196000e7, "R2": 0.9436},
    {"name": "CatBoost", "MSE": 3.933098e7, "R2": 0.9316},
    {"name": "LightGBM", "MSE": 5.410248e7, "R2": 0.9063},
    {"name": "Gradient Boosting", "MSE": 1.025993e8, "R2": 0.8242},
    {"name": "KNN", "MSE": 1.066068e8, "R2": 0.8179},
]

manual_model = {"name": "KNN_manual", "MSE": 1.77964266e8, "R2": 0.6979}

plt.figure(figsize=(10,6))

for m in automl_models:
    plt.scatter(m["MSE"], m["R2"], label=m["name"], s=80)

plt.scatter(manual_model["MSE"], manual_model["R2"], color="red", label=manual_model["name"], s=100, marker="X")

plt.xlabel("MSE")
plt.ylabel("R²")
plt.title("Сравнение моделей: AutoML vs ручная KNN")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
