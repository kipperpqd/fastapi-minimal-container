from fastapi import APIRouter

router = APIRouter()

@router.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Olá, {name}!"}
