import logging
import joblib
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, ValidationError, field_validator


class Apartment(BaseModel):
    area: int
    num_rooms: int
    total_floors: int
    floor: int

    @field_validator('area', 'num_rooms', 'total_floors', 'floor', mode='after')
    @classmethod
    def is_positive(cls, value: int):
        if value < 1:
            raise ValueError("Must be more than 0")

        return value



app = Flask(__name__)
model = joblib.load('models/linear_regression_model.pkl')

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

        prediction = model.predict([[area]])
        pretty_prediction = round((prediction[0] / 1_000_000), 2)
        
        return jsonify({"message": "Данные валидны", "prediction": pretty_prediction })
    except ValidationError as e:
        return e.json(), 422, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
