
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


energy_efficiency = fetch_ucirepo(id=242)

X = energy_efficiency.data.features
y = energy_efficiency.data.targets["Y1"]  # Heating Load — целевая переменная


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1337
)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }


# Базовая модель

base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())
])

base_model.fit(X_train, y_train)
base_results = evaluate(base_model, X_test, y_test)

print("\n=== Базовая модель (Linear Regression) ===")
print(base_results)

# L1-регуляризация (Lasso)

lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", Lasso())
])

params_lasso = {"reg__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

grid_lasso = GridSearchCV(lasso, params_lasso, cv=5)
grid_lasso.fit(X_train, y_train)

lasso_results = evaluate(grid_lasso.best_estimator_, X_test, y_test)

print("\n=== LASSO (L1) ===")
print("Лучшее alpha:", grid_lasso.best_params_)
print(lasso_results)

# L2-регуляризация (Ridge)

ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", Ridge())
])

params_ridge = {"reg__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

grid_ridge = GridSearchCV(ridge, params_ridge, cv=5)
grid_ridge.fit(X_train, y_train)

ridge_results = evaluate(grid_ridge.best_estimator_, X_test, y_test)

print("\n=== Ridge (L2) ===")
print("Лучшее alpha:", grid_ridge.best_params_)
print(ridge_results)

# ElasticNet (L1 + L2)

elastic = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", ElasticNet())
])

params_elastic = {
    "reg__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "reg__l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

grid_elastic = GridSearchCV(elastic, params_elastic, cv=5)
grid_elastic.fit(X_train, y_train)

elastic_results = evaluate(grid_elastic.best_estimator_, X_test, y_test)

print("\n=== ElasticNet ===")
print("Лучшие параметры:", grid_elastic.best_params_)
print(elastic_results)


results_table = pd.DataFrame({
    "Model": ["Linear Regression", "LASSO", "Ridge", "ElasticNet"],
    "MSE": [base_results["MSE"], lasso_results["MSE"], ridge_results["MSE"], elastic_results["MSE"]],
    "MAE": [base_results["MAE"], lasso_results["MAE"], ridge_results["MAE"], elastic_results["MAE"]],
    "R2": [base_results["R2"], lasso_results["R2"], ridge_results["R2"], elastic_results["R2"]],
})

print("\n=== Сравнение моделей ===")
print(results_table)
