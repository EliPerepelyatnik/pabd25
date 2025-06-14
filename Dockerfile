# Используем официальный python-образ
FROM python:3.13-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /service

# Копируем все файлы в контейнер
COPY requirements.txt .
COPY service/ /service/
COPY src/ /src/

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip && pip install -r requirements.txt

# Запускаем приложение через Gunicorn с одним воркером
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]