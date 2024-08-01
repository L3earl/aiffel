from fastapi import FastAPI
from pydantic import BaseModel
from calculator import calculate

app = FastAPI()

@app.get("/")
async def get_root_route():
    return {"message": "hello modu"}


class Formula(BaseModel):
    x: float
    y: float
    operator: str


@app.post("/calculator")
async def input_formula(input: Formula):
    result = calculate(x=input.x, y=input.y, operator=input.operator)
    return result