import pandas as pd
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor


def train_model(model_name):
    
    train_data = pd.read_csv('C:/Users/eliza/Projects/pabd25/data/processed/train.csv')

    X_train = train_data[
        ["total_meters", "floor", "floors_count", "rooms_count"]
    ]  # 4 признака
    y_train = train_data["price"]

    # Создание и обучение модели
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Сохраняем модель
    save_model(model, model_name)

def save_model(model, model_name):
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"Модель сохранена как: {model_path}")
    return model_path

def main():
    model_name = "model_gbr_2105"
    train_model(model_name)

if __name__ == "__main__":
    main()
    
        
