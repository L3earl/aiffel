# query_param.py
from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items/")
# Query Parameter 는 function parameter 로는 사용되지만 Path Operation 에 포함되지 않아
# Path Parameter 라고 할 수 없는 parameter 를 의미
# http://localhost:8000/items/?skip=0&limit=10
def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]


# needy 는 Path Operation @app.get("/items/{item_id}") 에 포함되어 있지 않으므로 
# Query Parameter 이고, function read_user_item() 에서 기본값이 존재하지 않기 때문에 
# Required Query Parameter 임
# http://localhost:8000/items/foo-item?needy=someneedy
@app.get("/items/{item_id}")
def read_user_item(item_id: str, needy: str):
    item = {"item_id": item_id, "needy": needy}
    return item