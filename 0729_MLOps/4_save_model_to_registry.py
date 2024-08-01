# db_train.py 에서 데이터 저장하는 부분을 띄어놓은 MLFLOW 서버에 저장하도록 변경 
# 학습이 끝난 모델을 MLflow 의 built-in method 를 사용해 MLflow 서버에 저장
    # 모델 저장 방법은 artifact, 빌트인 방법 2가지가 있음
# save_model_to_registry.py
import os
from argparse import ArgumentParser

import mlflow
import pandas as pd
import psycopg2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 0. set mlflow environments
# MLflow 와 통신하기 위해서는 몇 가지 환경 변수가 설정되어야 함
# 유저가 학습한 모델을 MLflow 서버를 통해 Artifact Store 인 MinIO 에 저장
# 이 과정에서 MinIO 의 접근 권한이 필요하게 됨
# 접근 권한 정보는 1) MLflow Setup 챕터의 Docker Compose 파일에서 설정한 mlflow-server , 
#   mlflow-artifact-store 의 정보와 같음
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000" # MinIO API 서버 주소
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001" # MLflow 서버 주소
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

# 1. get data
db_connect = psycopg2.connect(
    user="myuser",
    password="mypassword",
    host="localhost",
    port=5432,
    database="mydatabase",
)
df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db_connect)

X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 2. model development and train
model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
model_pipeline.fit(X_train, y_train)

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 3. save model
# MLflow 는 정보를 저장하기 위해 experiment 와 run 을 사용
# experiment : MLflow 에서 정보를 관리하기 위해 나누는 일종의 directory 입니다. 
#   BERT, ResNet 과 같이 특정 이름을 통해 생성 할 수 있으며, 생성하지 않고 MLflow 에 
#   정보를 저장하는 경우 Default 라는 이름의 experiment 에 저장됩니다.
# run : experiment 에 저장되는 모델 실험 결과 입니다. 
#   해당 run 에 실제 정보들이 저장되게 되며, experiment/run 의 구조로 저장됩니다.
# MLflow 는 정보 저장에 관련된 스크립트를 실행 할 때 명시된 experiment 에 run 을 동적으로 생성합니다. 
#   이 때, 각각의 run 은 unique 한 해쉬값인 run_id 를 부여받게 되며 이를 이용하여 저장된 후에도 
#   해당 정보에 접근할 수 있습


# 모델의 이름을 설정할 수 있는 외부 변수를 설정합니다. 
# MLflow 에서는 모델을 저장할 때 이름을 설정하여 관리하게 됩니다. 이번 챕터에서는 기본값으로 sk_model 을 사용
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="sk_model")
args = parser.parse_args()

# experiment 를 설정합니다. mlflow.set_experiment 함수는 experiment 가 존재하지 않는 경우 
# 새로 생성되며, 존재하는 경우 해당 experiment 를 사용합
mlflow.set_experiment("new-exp") # experiment 이름 지정하여 생성 

# 추후 잘못된 정보들이 들어올 경우 에러를 발생시키기 위해, 모델에 입력값 정보들을 설정 ?
signature = mlflow.models.signature.infer_signature(model_input=X_train, model_output=train_pred)
input_sample = X_train.iloc[:10]

# run 을 생성하고 정보를 저장
with mlflow.start_run():
    # 모델의 결과 metrics 를 Python 의 dictionary 형태로 입력해 생성된 run 에 저장
    mlflow.log_metrics({"train_acc": train_acc, "valid_acc": valid_acc})
    # sklearn 의 모델은 mlflow.sklearn 를 사용하여 간편하게 업로드가 가능
    # 학습된 모델 결과물이 sklearn 객체일 경우 [MLFlow Storage Format]의 구조로 run 에 저장
    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path=args.model_name,
        signature=signature,
        input_example=input_sample,
    )

# 4. save data
df.to_csv("data.csv", index=False)