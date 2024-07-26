#Dockerfile
FROM --platform=$BUILDPLATFORM python:3.10-alpine AS builder

WORKDIR /code

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python3"]
CMD ["app.py"]

FROM builder as dev-envs

RUN apk update && apk add --no-cache bash git shadow

RUN useradd -s /bin/bash -m vscode && \
    groupadd docker && \
    usermod -aG docker vscode

COPY --from=gloursdocker/docker / /