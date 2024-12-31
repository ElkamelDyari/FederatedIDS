from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
import uvicorn
from backend.utils.DataPreparation import load_data
from backend.utils.LoadModel import retrieve_latest_registered_model

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = retrieve_latest_registered_model("Federated Learning IDS")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        processed_data = load_data(df)
        attack_type_mapping = {
            0: "BENIGN",
            1: "Bot",
            2: "Brute Force",
            3: "DoS",
            4: "Heartbleed",
            5: "Infiltration",
            6: "Port Scan",
            7: "Web Attack"
        }
        predictions = model.predict(processed_data)
        attack_type_predictions = [attack_type_mapping[pred] for pred in predictions]
        return {"predictions": attack_type_predictions}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
