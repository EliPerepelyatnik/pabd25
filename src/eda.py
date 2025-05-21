import pandas as pd
import os
import glob 

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

    # Ограничение по цене — 85-й процентиль
    q = df["price"].quantile(0.85)
    print(f"85-й процентиль по цене: {q}")
    df = df[df["price"] <= q]

    df["flat_id"] = df["url"].str.extract(r"/flat/(\d+)/")[0].astype("int64")

    test_size = int(len(df["flat_id"]) * 0.2)
    test_items = sorted(df["flat_id"])[-test_size:]

    train_data = df[~df["flat_id"].isin(test_items)]
    test_data = df[df["flat_id"].isin(test_items)]

    os.makedirs('C:/Users/eliza/Projects/pabd25/data/processed', exist_ok=True)
    train_data.to_csv('C:/Users/eliza/Projects/pabd25/data/processed/train.csv', encoding='utf-8')
    test_data.to_csv('C:/Users/eliza/Projects/pabd25/data/processed/test.csv', encoding='utf-8')

def main():
    preprocess_data()

if __name__ == "__main__":
    main()