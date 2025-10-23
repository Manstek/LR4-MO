import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from Dataset_download import download_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.optimize import minimize

from tensorflow import keras

warnings.filterwarnings("ignore")


def plot_predictions(test, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(test, predicted, alpha=0.5)
    plt.plot([test.min(), test.max()], [test.min(), test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()


# 1. Модель LM (Linear Model)
def lm(data_train, data_test, formula, y_test, graph_desc):
    lm_model = smf.ols(formula, data_train).fit()
    predictions = lm_model.predict(data_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    plot_predictions(y_test, predictions,
                     'Comparison of Actual vs Predicted for ' + graph_desc)
    return {"r2": r2, "mse": mse, "mae": mae}


# 2. Модель GLM (Generalized Linear Model)
def glm(df_train, df_test, y_train, y_test, formula, graph_desc):
    glm_model = smf.glm(formula, df_train, family=sm.families.Gaussian())
    result = glm_model.fit()
    predictions = result.predict(df_test)
    plot_predictions(y_test, predictions,
                     'Comparison of Actual vs Predicted for ' + graph_desc)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"r2": r2, "mse": mse, "mae": mae}


# 3. Модель NLM (Nonlinear Model)
def objective(data1, data2):
    return np.dot(data2, data1)


def percentile(percentile_, data, response):
    return np.sum((objective(percentile_, data) - response) ** 2)


def nlm(x_train, x_test, y_train, y_test, graph_desc):
    min_x = minimize(percentile, np.ones(8), args=(x_train, y_train)).x
    predictions = objective(min_x, x_test)
    plot_predictions(y_test, predictions,
                     'Comparison of Actual vs Predicted for ' + graph_desc)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"r2": r2, "mse": mse, "mae": mae}


# 4. Нейросетевая модель (Neural Model)
def nm(x_train, x_test, y_train, y_test, graph_desc):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(
            x_train_scaled.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(x_train_scaled, y_train, epochs=100, batch_size=10,
              validation_split=0.1, verbose=0)

    predictions = model.predict(x_test_scaled).flatten()
    plot_predictions(y_test, predictions,
                     'Comparison of Actual vs Predicted for ' + graph_desc)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {"r2": r2, "mse": mse, "mae": mae}


# 5. Медианная регрессия с MAPE
def mape(train_x, test_x, train_y, test_y, graph_desc, use_mape_weights=False):
    eps = 1e-8

    qr = QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs')

    if use_mape_weights:
        sample_weight = 1.0 / (np.abs(train_y) + eps)
        qr.fit(train_x, train_y, sample_weight=sample_weight)
    else:
        qr.fit(train_x, train_y)

    predictions = qr.predict(test_x)

    plot_predictions(test_y, predictions,
                     'Comparison of Actual vs Predicted for ' + graph_desc)

    mae = mean_absolute_error(test_y, predictions)
    mse = mean_squared_error(test_y, predictions)
    mape_val = np.mean(
        np.abs((test_y - predictions) / (np.abs(test_y) + eps))) * 100

    return {"mape": mape_val, "mse": mse, "mae": mae}


if __name__ == "__main__":

    dataset_path = download_dataset("datasets")
    if dataset_path:
        df = pd.read_excel(dataset_path)
        print("Размер датасета:", df.shape)
        print("\nСам датасет:")
        print(df)
    else:
        raise RuntimeError("Не удалось скачать/найти датасет")

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=1337)

    # Подготовка данных для Y1 (heating)
    X_ml_train_h = X_train[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y_ml_train_h = X_train['Y1']
    X_ml_test_h = X_test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y_ml_test_h = X_test['Y1']

    # Подготовка данных для Y2 (cooling)
    X_ml_train_c = X_train[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y_ml_train_c = X_train['Y2']
    X_ml_test_c = X_test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y_ml_test_c = X_test['Y2']

    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    print("\n" + "=" * 50)
    print("МОДЕЛЬ LM")
    print("=" * 50)

    # Построение моделей LM для Y1 и Y2
    formula = "Y1 ~ " + " + ".join(df.columns.drop(["Y1", "Y2"]))
    lm_h = lm(X_train, X_test, formula, y_ml_test_h, 'Y1 - LM')

    formula = "Y2 ~ " + " + ".join(df.columns.drop(["Y1", "Y2"]))
    lm_c = lm(X_train, X_test, formula, y_ml_test_c, 'Y2 - LM')

    print("LM для Y1:", lm_h)
    print("LM для Y2:", lm_c)

    print("\n" + "=" * 50)
    print("МОДЕЛЬ GLM")
    print("=" * 50)

    # Построение моделей GLM для Y1 и Y2
    formula_h = "Y1 ~ X1+X2+X3+X4+X5+X6+X7+X8"
    glm_h = glm(X_train, X_test, y_ml_train_h, y_ml_test_h, formula_h,
                'Y1 - GLM')

    formula_c = "Y2 ~ X1+X2+X3+X4+X5+X6+X7+X8"
    glm_c = glm(X_train, X_test, y_ml_train_c, y_ml_test_c, formula_c,
                'Y2 - GLM')

    print("GLM для Y1:", glm_h)
    print("GLM для Y2:", glm_c)

    print("\n" + "=" * 50)
    print("МОДЕЛЬ NLM")
    print("=" * 50)

    # Построение моделей NLM для Y1 и Y2
    nlm_h = nlm(X_ml_train_h.to_numpy(), X_ml_test_h.to_numpy(),
                y_ml_train_h.to_numpy(), y_ml_test_h.to_numpy(), 'Y1 - NLM')

    nlm_c = nlm(X_ml_train_c.to_numpy(), X_ml_test_c.to_numpy(),
                y_ml_train_c.to_numpy(), y_ml_test_c.to_numpy(), 'Y2 - NLM')

    print("NLM для Y1:", nlm_h)
    print("NLM для Y2:", nlm_c)

    print("\n" + "=" * 50)
    print("НЕЙРОСЕТЕВАЯ МОДЕЛЬ")
    print("=" * 50)

    # Построение нейросетевых моделей для Y1 и Y2
    nm_h = nm(X_ml_train_h, X_ml_test_h, y_ml_train_h, y_ml_test_h,
              'Y1 - Neural Network')
    nm_c = nm(X_ml_train_c, X_ml_test_c, y_ml_train_c, y_ml_test_c,
              'Y2 - Neural Network')

    print("Neural Network для Y1:", nm_h)
    print("Neural Network для Y2:", nm_c)

    print("\n" + "=" * 50)
    print("МЕДИАННАЯ РЕГРЕССИЯ (MAPE)")
    print("=" * 50)

    # Медианная регрессия для Y1 и Y2 (классическая L1)
    mape_h = mape(X_ml_train_h, X_ml_test_h, y_ml_train_h, y_ml_test_h,
                  'Y1 - Quantile Regression')
    mape_c = mape(X_ml_train_c, X_ml_test_c, y_ml_train_c, y_ml_test_c,
                  'Y2 - Quantile Regression')

    print("Quantile Regression для Y1:", mape_h)
    print("Quantile Regression для Y2:", mape_c)

    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)

    results = {
        'Model': ['LM', 'GLM', 'NLM', 'Neural Network', 'Quantile Regression'],
        'Y1_R2': [lm_h['r2'], glm_h['r2'], nlm_h['r2'], nm_h['r2'], '-'],
        'Y1_MSE': [lm_h['mse'], glm_h['mse'], nlm_h['mse'], nm_h['mse'], mape_h['mse']],
        'Y1_MAE': [lm_h['mae'], glm_h['mae'], nlm_h['mae'], nm_h['mae'], mape_h['mae']],
        'Y1_MAPE': ['-', '-', '-', '-', mape_h['mape']],
        'Y2_R2': [lm_c['r2'], glm_c['r2'], nlm_c['r2'], nm_c['r2'], '-'],
        'Y2_MSE': [lm_c['mse'], glm_c['mse'], nlm_c['mse'], nm_c['mse'], mape_c['mse']],
        'Y2_MAE': [lm_c['mae'], glm_c['mae'], nlm_c['mae'], nm_c['mae'], mape_c['mae']],
        'Y2_MAPE': ['-', '-', '-', '-', mape_c['mape']]
    }

    results_df = pd.DataFrame(results)
    print(results_df.round(4))

    # Определение лучшей модели для Y1
    y1_models = ['LM', 'GLM', 'NLM', 'Neural Network']
    y1_r2_scores = [lm_h['r2'], glm_h['r2'], nlm_h['r2'], nm_h['r2']]
    best_y1_model = y1_models[np.argmax(y1_r2_scores)]

    # Определение лучшей модели для Y2
    y2_models = ['LM', 'GLM', 'NLM', 'Neural Network']
    y2_r2_scores = [lm_c['r2'], glm_c['r2'], nlm_c['r2'], nm_c['r2']]
    best_y2_model = y2_models[np.argmax(y2_r2_scores)]

    print(f"\nЛучшая модель для Y1 (Heating Load): {best_y1_model} (R² = {max(y1_r2_scores):.4f})")
    print(f"Лучшая модель для Y2 (Cooling Load): {best_y2_model} (R² = {max(y2_r2_scores):.4f})")

    # Визуализация сравнения моделей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Сравнение R2 для Y1
    models = ['LM', 'GLM', 'NLM', 'Neural Network']
    y1_scores = [lm_h['r2'], glm_h['r2'], nlm_h['r2'], nm_h['r2']]
    y2_scores = [lm_c['r2'], glm_c['r2'], nlm_c['r2'], nm_c['r2']]

    ax1.bar(models, y1_scores, color=['blue', 'green', 'orange', 'red'])
    ax1.set_title('Сравнение R² для Y1 (Heating Load)')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)

    ax2.bar(models, y2_scores, color=['blue', 'green', 'orange', 'red'])
    ax2.set_title('Сравнение R² для Y2 (Cooling Load)')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
