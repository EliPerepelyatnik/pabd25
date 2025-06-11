import logging
import joblib
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, ValidationError, field_validator
from flask_cors import CORS
from dotenv import load_dotenv
import os
from flask_httpauth import HTTPTokenAuth
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import io


# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("BUCKET_NAME")
AWS_S3_MODEL_KEY = os.getenv("LOCAL_FILE_PATH")
CORRECT_TOKEN=os.getenv("API_TOKEN")


class Apartment(BaseModel):
    area: int
    num_rooms: int
    total_floors: int
    floor: int
    token: str

    @field_validator('area', 'num_rooms', 'total_floors', 'floor', mode='after')
    @classmethod
    def is_positive(cls, value: int):
        if value < 1:
            raise ValueError("Must be more than 0")

        return value



app = Flask(__name__)
CORS(app)
auth = HTTPTokenAuth(scheme="Bearer")

@auth.verify_token
def verify_token(token):
    return token == CORRECT_TOKEN


def load_model_from_s3():
    try:
        s3 = boto3.client(
            's3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
        
        # Получаем объект из S3
        response = s3.get_object(Bucket=AWS_S3_BUCKET, Key='perepelyatnik/models/gbr_model_v15.pkl')
        
        return joblib.load(io.BytesIO(response['Body'].read()))
        
           
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        raise
    except ClientError as e:
        logger.error(f"S3 error: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise


# Инициализация модели при старте приложения
try:
    model = load_model_from_s3()
    logger.info("Модель успешно загружена из S3")
except Exception as e:
    logger.critical(f"Не удалось загрузить модель: {e}")
    exit(1)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    try:
        json_ = request.json
        apartment = Apartment.model_validate(json_)
        area = apartment.area
        num_rooms = apartment.num_rooms
        total_floors = apartment.total_floors
        floor = apartment.floor

        if apartment.token != CORRECT_TOKEN:
            logger.warning("Неверный токен")
            return jsonify({"error": "Неверный токен"}), 403

        prediction = model.predict([[area, num_rooms, total_floors, floor]])
        pretty_prediction = round((prediction[0] / 1_000_000), 2)
        
        return jsonify({"message": "Данные валидны", "prediction": pretty_prediction })
    except ValidationError as e:
        return e.json(), 422, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
