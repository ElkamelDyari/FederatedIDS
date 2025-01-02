import numpy as np
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
import uvicorn
#from backend.utils.DataPreparation import data_cleaning #use this for running from root
#from backend.utils.LoadModel import retrieve_latest_registered_model #use this for running from root

from utils.DataPreparation import data_cleaning

from utils.LoadModel import retrieve_latest_registered_model


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
# Get the input signature
input_signature = model.metadata.get_input_schema()
number_required_columns = len(input_signature)
# Print the number of required fields
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        col_to_show = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s"]
        data_without_duplicates = df.drop_duplicates()
        data_without_duplicates = data_without_duplicates[col_to_show]
        processed_data = data_cleaning(df, n_featues=number_required_columns)
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
        data_without_duplicates["Attack Type"] = attack_type_predictions
        data_without_duplicates.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_without_duplicates['Flow Bytes/s'].fillna("NAN", inplace=True)
        data_without_duplicates['Flow Packets/s'].fillna("NAN", inplace=True)

        return {"predictions": data_without_duplicates.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)