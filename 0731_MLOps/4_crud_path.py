# path_param을 이용하여 CRUD (생성, 읽기, 업데이트, 삭제)를 구현 
# crud_path.py
from fastapi import FastAPI, HTTPException

# Create a FastAPI instance
app = FastAPI()

# User database
USER_DB = {}

# Fail response
# 메모리에 존재하지 않는 이름에 대한 요청이 들어온 경우에 에러를 발생할 수 있도록 HTTPException 을 이용하여 NAME_NOT_FOUND 를 선언
NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found.")


@app.post("/users/name/{name}/nickname/{nickname}")
def create_user(name: str, nickname: str):
    USER_DB[name] = nickname
    return {"status": "success"}


@app.get("/users/name/{name}")
def read_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    return {"nickname": USER_DB[name]}


@app.put("/users/name/{name}/nickname/{nickname}")
def update_user(name: str, nickname: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    USER_DB[name] = nickname
    return {"status": "success"}


@app.delete("/users/name/{name}")
def delete_user(name: str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    del USER_DB[name]
    return {"status": "success"}