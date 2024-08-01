# path_param.py
from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# Path Parameter 는 Path Operation 에 포함된 변수로 사용자에게 
# 입력받아 function 의 argument 로 사용되는 parameter 를 의미
# http://localhost:8000/items/1
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}