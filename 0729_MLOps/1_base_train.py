# base_train.py 학습 및 데이터 저장
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. get data
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2022)

# 2. model development and train
# 데이터를 scaling 하기 위해 sklearn.preprocessing 의 StandardScaler 를 scaler 에 할당
scaler = StandardScaler()
classifier = SVC()

# fit 을 통해 scaler를 학습한 후 transform을 이용해 데이터를 scaling 합니다.
    # fit 메소드는 데이터를 이용하여 변환에 필요한 통계치를 계산합니다. 
        # 예를 들어, StandardScaler의 경우, fit 메소드는 데이터의 평균과 표준편차를 계산
    # transform 메소드는 fit에서 계산된 통계치를 사용하여 데이터를 변환
scaled_X_train = scaler.fit_transform(X_train) # 학습데이터는 통계치 계산 + 스케일링 변환
scaled_X_valid = scaler.transform(X_valid) # 검증데이터는 같은 기준으로 스케일링 변환

# scaling 전 데이터와 scaling 후 데이터를 비교하면 다음과 같습니다.
print(X_train.values[0])
print(scaled_X_train[0])

# 모델 학습
classifier.fit(scaled_X_train, y_train)

train_pred = classifier.predict(scaled_X_train)
valid_pred = classifier.predict(scaled_X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 3. save model
joblib.dump(scaler, "scaler.joblib")
joblib.dump(classifier, "classifier.joblib")