from fastapi import FastAPI
import datetime

app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}
