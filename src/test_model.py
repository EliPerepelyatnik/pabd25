import pandas as pd
import os
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def test_model(model_name):
    
    test_path = 'C:/Users/eliza/Projects/pabd25/data/processed/test.csv'
    model_path = f'models/{model_name}.pkl'
    
    # Загрузка модели
    model = joblib.load(model_path)
    print(f"Модель загружена из: {model_path}")

    # Загрузка тестовых данных
    test_data = pd.read_csv(test_path)
    
    X_test = test_data[
        ["total_meters", "floor", "floors_count", "rooms_count"]
    ]  # 4 признака
    y_test = test_data["price"]

    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Вывод метрик качества
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    print(f"Коэффициент детерминации R²: {r2:.6f}")
    print(f"Средняя ошибка предсказания: {np.mean(np.abs(y_test - y_pred)):.2f} рублей")
   
        
def main():
    model_name = "model_gbr_2105"

    test_model(model_name=model_name)

if __name__ == "__main__":
    main()