# !pip install fastapi uvicorn
from fastapi import FastAPI
from fastapi import APIRouter

# app = FastAPI(docs_url=None)
app = FastAPI()

@app.get("/")
def read_root():
    return("Hey")

@app.get("/home/{name}")
def health_check_handler(name: str):
    return {"name": name}

@app.get("/home_error/{x}")
def health_check_handler(x: int):
    return {"x": x}

items_db = [{"item_name":"fooo"}, {"test":"pizza"}]

@app.get("/items/{items_id}")
def read_item(item_id: str, skip:int=0, limit:int=10):
    return items_db[skip:skip+limit]

@app.post("/")
def home_post(msg:str):
    return {"hello": "post", "ma" : msg}

router = APIRouter()
@router.get("hello")
async def say_hello() -> dict:
    return {"message":'d'}


from pydantic import BaseModel

class HappyBook(BaseModel):
    id: int
    Name: int
    publishers: str
    sbn: str

from enum import Enum
from fastapi import FastAPI


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "modumodufighting"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "lenet jjange"}

    return {"model_name": model_name, "message": "Have good!"}



from typing import List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

class Movie(BaseModel):
    mid: int
    genre: str
    rate: Union[int, float, str] # Union 안에 있는 자료형이 모두 올 수 있음
    tag: Optional[str] = None # 기본값 설정하는 방법
    date: Optional[datetime] = None
    some_variable_list: List[int] = []

class User(BaseModel):
    uid: int
    name : str = Field(min_length=1, max_length=103) # Field 데이터의 범위를 제한하는 기능
    age : int = Field(gt=1, le=130 ) # greater then, less then

tmp_data = {
    "mid": '1',
    "genre": "action",
    "rate": 1.5,
    "tag": None,
    "date": "2024-07-31 12:22:11",
}

tmp_user_data = {"uid": 100, "name":"sdd", "age":11}

tmp_movie = Movie(**tmp_data)
tmp_user = User(**tmp_user_data)
print(tmp_movie)
print(tmp_user)