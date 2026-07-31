FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "validator.run", "--config", "/config/validator.yaml", "--continuous", "--interval", "60"]
