# Nothing needs to be installed on your machine except Docker itself --
# this image is where requirements.txt actually gets installed.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.web:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
