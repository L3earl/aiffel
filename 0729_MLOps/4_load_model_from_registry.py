# MLflow 에 저장된 모델을 불러올 수 있는 스크립트를 작성합니다.
# 불러온 모델을 통해 추론하고 결과를 확인합니다.
# load_model_from_registry.py
# 학습이 끝난 모델을 MLflow built-in method 를 사용하여 MLflow 서버에서 불러옵니다
import os
from argparse import ArgumentParser

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 0. set mlflow environments
# Save Model to Registry 챕터와 같이 MLflow 서버에 접근하기 위한 환경 변수를 설정
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 1. load model from mlflow
# mlflow.sklearn.load_model 함수를 사용하여 저장된 모델을 불러옵니다. 
# 모델을 포함하고 있는 run_id 와 모델을 저장할 때 설정했던 모델 이름을 받을 수 있도록 외부 변수를 설정
parser = ArgumentParser()
parser.add_argument("--run-id", dest="run_id", type=str, default="ab4b5b8102d04a4cb6493074b5b4c511")
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
args = parser.parse_args()

# 앞에서 받은 변수와 sklearn의 load_model을 사용해서 불러옴 
model_pipeline = mlflow.sklearn.load_model(f"runs:/{args.run_id}/{args.model_name}")
print(model_pipeline)

# pyfunc 방식으로 모델 불러오기
# MLflow 에서는 지정한 방식 [MLFlow Storage Format]에 따라 저장되어있는 모델에 대해서는 
# 종류에 관계없이 mlflow.pyfunc.load_model 을 이용하여 쉽게 모델을 불러올 수 있습니다.
model_pipeline = mlflow.pyfunc.load_model(f"runs:/{args.run_id}/{args.model_name}")
print(model_pipeline)

# 추론 코드 작성하기

# 2. get data
# Save Model to Registry 챕터에서 저장했던 데이터인 data.csv 파일로부터 데이터를 불러옵니다.
df = pd.read_csv("data.csv")

# 학습 조건과 같도록 불필요한 columns 를 제거하고, 학습 데이터와 평가 데이터로 분리
X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 3. predict results
train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)