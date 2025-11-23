from fastapi import FastAPI
from pydantic import BaseModel
import datetime

from app.routes.hello import router as hello_router  # <-- IMPORTANTE

app = FastAPI()

# Registrar as rotas
app.include_router(hello_router)

class TaskRequest(BaseModel):
    nome: str
    parametros: dict | None = None


@app.get("/ping")
def ping():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}


@app.post("/run-task")
def run_task(payload: TaskRequest):
    """
    Exemplo de função que pode ser chamada pelo n8n.
    Você coloca qualquer lógica Python aqui.
    """
    return {
        "mensagem": f"Tarefa '{payload.nome}' executada com sucesso!",
        "parametros_recebidos": payload.parametros,
        "executado_em": datetime.datetime.now().isoformat(),
    }
