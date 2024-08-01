#Dockerfile
FROM --platform=$BUILDPLATFORM python:3.12-slim AS builder

WORKDIR /code

COPY . .

RUN pip install poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]