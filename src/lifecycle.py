import datetime
import cianparser
import pandas as pd
import os
import glob 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

"""  Parse data from cian.ru
https://github.com/lenarsaitov/cianparser
"""

def parse_cian():
    moscow_parser = cianparser.CianParser(location="Москва")
    all_data = []
    room_variants = [1, 2, 3]

    for n_rooms in room_variants:
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 16,
                "end_page": 21,
                "object_type": "secondary"
            }
        )
        all_data.extend(data)

    df_new = pd.DataFrame(all_data)
    df_new['parse_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    room_str = "-".join(str(r) for r in room_variants)
    csv_path = f'data/raw/{room_str}_flats.csv'


    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates()
        df_combined.to_csv(csv_path, encoding='utf-8', index=False)
    else:
        df_new.to_csv(csv_path, encoding='utf-8', index=False)


def preprocess_data():
  
    raw_data_path = 'data/raw'
    file_list = glob.glob(raw_data_path + "/*.csv")
    cwd = os.getcwd()
    print(cwd)
    print(file_list) 

    df = pd.read_csv(file_list[0])
    print("Столбцы до удаления:", df.columns.tolist())
    
    df.drop(
        [
            "author",
            "author_type",
            "location",
            "deal_type",
            "accommodation_type",
            "price_per_month",
            "commissions",
            "district",
            "street",
            "house_number",
            "underground",
            "residential_complex",
            "parse_date",
        ],
        axis=1,
        inplace=True,
    )
    print("Столбцы после удаления:", df.columns.tolist())
    
    # Удаление строк с пропусками
    print("Количество пропусков до dropna:", df.isna().sum().sum())
    df = df.dropna()
    print("Количество пропусков после dropna:", df.isna().sum().sum())
    
    # Ограничение по площади
    df = df[df["total_meters"] <= 100]

    # Ограничение по цене — 99-й процентиль
    q = df["price"].quantile(0.85)
    print(f"85-й процентиль по цене: {q}")
    df = df[df["price"] <= q]

    os.makedirs('C:/Users/eliza/Projects/pabd25/data/processed', exist_ok=True)
    df.to_csv('C:/Users/eliza/Projects/pabd25/data/processed/data_cleaned.csv', encoding='utf-8')


def train_test_model(model_name):
    new_df = pd.read_csv("C:/Users/eliza/Projects/pabd25/data/processed/data_cleaned.csv")
    new_df["flat_id"] = new_df["url"].str.extract(r"/flat/(\d+)/")[0].astype("int64")
    print(new_df.columns)

    test_size = int(len(new_df["flat_id"]) * 0.2)
    test_items = sorted(new_df["flat_id"])[-test_size:]

    train_data = new_df[~new_df["flat_id"].isin(test_items)]
    test_data = new_df[new_df["flat_id"].isin(test_items)]
    
    X_train = train_data[
        ["total_meters", "floor", "floors_count", "rooms_count"]
    ]  # 4 признака
    y_train = train_data["price"]

    X_test = test_data[
        ["total_meters", "floor", "floors_count", "rooms_count"]
    ]  # 4 признака
    y_test = test_data["price"]


    # Создание и обучение модели
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Вывод метрик качества
    # todo: use logging
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    print(f"Коэффициент детерминации R²: {r2:.6f}")
    print(f"Средняя ошибка предсказания: {np.mean(np.abs(y_test - y_pred)):.2f} рублей")

    # Сохраняем модель под заданным именем
    save_model(model, model_name)

def save_model(model, model_name):
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"Модель сохранена как: {model_path}")
    return model_path
    
        
def main():
    # необходимо ввести имя модели вручную
    model_name = input("Введите имя модели для сохранения (например: gbr_model_v1): ").strip()

    print("1. Парсинг данных...")
    parse_cian()
    
    print("2. Предобработка данных...")
    preprocess_data()
    
    print("3. Обучение модели...")
    train_test_model(model_name=model_name)

if __name__ == "__main__":
    main()

