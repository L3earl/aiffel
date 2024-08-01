#MLflow 서버에 필요한 패키지가 설치된 이미지를 build 할 Dockerfile 을 작성합니다.
#서버를 띄울 때, MinIO 에 초기 bucket 을 생성하기 위해 MinIO Client 도 함께 설치합니다.
FROM amd64/python:3.9-slim

#git 은 MLflow 서버 내부 동작에, wget 은 MinIO Client 를 설치하기 위해 사용
RUN apt-get update && apt-get install -y \

    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

#MLflow 를 비롯해 PostgreSQL DB, AWS S3 에 관련된 Python 패키지를 설치
RUN pip install -U pip &&\

    pip install boto3==1.26.8 mlflow==1.30.0 psycopg2-binary

#앞서 설치한 wget 을 활용하여 MinIO Client 를 설치
RUN cd /tmp && \

    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/bin/mc