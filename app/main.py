from fastapi import FastAPI, Request
import datetime

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    print("Recebido do n8n:", payload)
    return {"status": "ok", "received": payload}

@app.post("/inserir_cliente")
async def inserir_cliente(request: Request):
    data = await request.json()
    # aqui você pode validar, processar, transformar
    return {"status": "cliente recebido", "dados": data}

@app.get("/clientes")
async def listar_clientes():
    return [
        {"id": 1, "nome": "Luis"},
        {"id": 2, "nome": "Reuben"}
    ]
