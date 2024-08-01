# multi_param.py
from typing import Union

from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# http://localhost:8000/users/3/items/foo-item?q=hello&short=True
# q 와 short 는 Query Parameter 임을 알 수 있고, 기본값이 각각 None 과 False 임
@app.get("/users/{user_id}/items/{item_id}")
def read_user_item(user_id: int, item_id: str, q: Union[str, None] = None, short: bool = False):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"},
        )
    return item