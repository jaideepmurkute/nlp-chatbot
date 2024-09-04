FROM python:3.12.4-slim AS base
# FROM python:3.12.4-alpine AS base

ENV FLASK_APP=src/app.py
ENV FLASK_ENV=development

WORKDIR /app

COPY requirements.txt /app/

# install deps with cleanup
# when using debian image like python:3.12.4-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc python3-dev \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# when using alpine image like python:3.12.4-alpine
# RUN apk add --no-cache \
#     gcc \
#     musl-dev \
#     libffi-dev \
#     && pip install --no-cache-dir -r requirements.txt \
#     && apk del gcc musl-dev libffi-dev \
#     && rm -rf /root/.cache

COPY src /app/src

EXPOSE 5100

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5100"]

