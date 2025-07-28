import os
    import torch
    import pickle
    import numpy as np
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List

    MODEL_PATH = os.environ.get("MODEL_PATH", "dl_ts_model_20250725_221216.pt")
    SCALER_X_PATH = os.environ.get("SCALER_X_PATH", "scaler_x.pkl")
    SCALER_Y_PATH = os.environ.get("SCALER_Y_PATH", "scaler_y.pkl")

    def load_scaler(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    scaler_x = load_scaler(SCALER_X_PATH)
    scaler_y = load_scaler(SCALER_Y_PATH)

    app = FastAPI()
    class PredictRequest(BaseModel):
        data: List[List[float]]
    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            X = np.array(req.data)
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = scaler_x.transform(X_flat).reshape(X.shape)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(X_tensor)
                y_pred = out.cpu().numpy()
            y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).squeeze().tolist()
            return {"result": y_pred_inv}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    