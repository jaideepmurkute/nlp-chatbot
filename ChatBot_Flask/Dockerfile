FROM python:3.12.4-slim AS base

ENV FLASK_APP=app.py
ENV FLASK_ENV=development

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5100

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5100"]

