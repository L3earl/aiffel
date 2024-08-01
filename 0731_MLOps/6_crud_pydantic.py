# 앞서 작성한 API 에서 CRUD 부분을 Pydantic 을 이용하여 수정
# crud_pydantic.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# CreateIn 에서는 이름과 별명을 입력받을 수 있도록 name 과 nickname 변수를 만듭니다.
class CreateIn(BaseModel):
    name: str
    nickname: str

# CreateOut 에는 status 와 id 변수를 만들어 Create 의 operation function 에서 
#   두 변수의 값을 return 하도록 합니다.
# id 는 create 되는 시점의 memory 에 존재하는 데이터의 개수로 정의하여 작성합니다
class CreateOut(BaseModel):
    status: str
    id: int

# Create a FastAPI instance
app = FastAPI()

# User database
USER_DB = {}

# Fail response # 잘못된 입력에 에러 발생 하도록
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found.")

# Request Body 는 client 에서 API 로 전송하는 데이터를 의미합니다. 
#   반대로 Response Body 는 API 가 client 로 전송하는 데이터를 의미합니다.
# 이렇게 client 와 API 사이에 데이터를 주고 받을 때 데이터의 형식을 지정해 줄 수 있는데, 
#   이를 위해 Pydantic Model 을 사용할 수 있습니다.

# Response Model
# @app.get(), @app.post() 등 다양한 Path Operation 에 response_model 을 이용하여 Response Body 에 사용될 데이터 모델을 지정해줄 수 있습니다. 또한, output data 의 type 을 체크하여 자동으로 변환시키고, type 이 유효한지 확인해주고, response 를 위해 자동으로 JSON Schema 를 추가해주는 등의 역할을 할 수 있습니다.
# 그 중에서도 response_model 의 가장 중요한 역할은 output data 의 형태를 제한해 줄 수 있다는 것입니다. 예를 들면, response_model=CreateOut 과 같이 지정해주면 해당 Path Operation 이 실행되었을 때 CreateOut 에 존재하는 attribute 의 형태로 데이터를 반환하게 됩니다. 이를 통해, Create API 에 입력하는 데이터로는 CreateIn 모델을, 반환하는 데이터로는 CreateOut 모델을 사용하도록 지정할 수 있습니다.
@app.post("/users", response_model=CreateOut)
# 파라미터 : user, type : CreateIn
def create_user(user: CreateIn):
    # Pydantic Model 에 선언된 변수를 사용하여 DB 에 사용자 정보를 저장
    USER_DB[user.name] = user.nickname
    user_dict = user.dict()
    user_dict["status"] = "success"
    user_dict["id"] = len(USER_DB)
    return user_dict
# Response Body 에 필요한 변수는 response_model 로 지정된 CreateOut 모델의 변수인 status 와 id 이기 때문에 이 변수들의 값을 저장해 주어야 합니다. status 와 id 값을 준 CreateOut 의 객체를 return 하면 됩니다.
# 정리하면, create_user 함수는 입력으로 CreateIn 모델을 받고, CreateOut 모델을 반환함으로써 request 할 때와 response 할 때 주고받는 데이터에 다른 변수를 사용할 수 있는 것입니다.


@app.get("/users")
def read_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    return {"nickname": USER_DB[name]}


@app.put("/users")
def update_user(name: str, nickname: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    USER_DB[name] = nickname
    return {"status": "success"}


@app.delete("/users")
def delete_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    del USER_DB[name]
    return {"status": "success"}