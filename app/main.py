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
