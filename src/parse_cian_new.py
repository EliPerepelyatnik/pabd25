import datetime
import cianparser
import pandas as pd
import os


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
                "start_page": 20,
                "end_page": 22,
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
    
def main():
    parse_cian()

if __name__ == "__main__":
    main()