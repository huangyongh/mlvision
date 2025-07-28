import streamlit as st
import os
import shutil


def run():


    st.header("🚀 一键自动部署API服务")

    # 1. 支持上传模型权重文件
    uploaded_file = st.file_uploader("上传模型权重文件（.pt 或 .pkl）", type=["pt", "pkl"])
    if uploaded_file is not None:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"已上传模型文件：{uploaded_file.name}")

    # 2. 识别模型类型
    model_files = [f for f in os.listdir(".") if f.endswith(".pt") or f.endswith(".pkl")]
    model_type = None
    model_file = None
    for f in model_files:
        if f.endswith(".pt"):
            model_type = "dl"
            model_file = f
            break
        elif f.endswith(".pkl"):
            model_type = "ml"
            model_file = f
            break
    if not model_type:
        st.error("未检测到模型文件（.pt 或 .pkl），请先上传或导出模型到当前目录！")
        st.stop()

    st.success(f"检测到模型类型：{'深度学习' if model_type=='dl' else '机器学习'}，文件：{model_file}")

    # 3. 生成API接口文件内容
    if model_type == "dl":
        api_code = '''import os
    import torch
    import pickle
    import numpy as np
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List

    MODEL_PATH = os.environ.get("MODEL_PATH", "{model_file}")
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
    '''.replace('{model_file}', model_file)
    else:
        api_code = '''import os
    import pickle
    import numpy as np
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List

    MODEL_PATH = os.environ.get("MODEL_PATH", "{model_file}")
    SCALER_X_PATH = os.environ.get("SCALER_X_PATH", "scaler_x.pkl")
    SCALER_Y_PATH = os.environ.get("SCALER_Y_PATH", "scaler_y.pkl")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_X_PATH, "rb") as f:
        scaler_x = pickle.load(f)
    with open(SCALER_Y_PATH, "rb") as f:
        scaler_y = pickle.load(f)

    app = FastAPI()
    class PredictRequest(BaseModel):
        data: List[List[float]]
    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            X = np.array(req.data)
            X_scaled = scaler_x.transform(X)
            y_pred = model.predict(X_scaled)
            y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).squeeze().tolist()
            return {"result": y_pred_inv}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    '''.replace('{model_file}', model_file)

    # 4. 生成Dockerfile内容
    api_file = "ml_api.py" if model_type=="ml" else "dl_api.py"
    dockerfile_code = f'''FROM python:3.9-slim\nWORKDIR /app\nCOPY . .\nRUN pip install fastapi uvicorn pydantic numpy torch scikit-learn\nEXPOSE 8000\nCMD [\"uvicorn\", \"{api_file.replace('.py','')}:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n'''

    # 5. 一键写入
    if st.button("一键生成API接口和Dockerfile"):
        with open(api_file, "w", encoding="utf-8") as f:
            f.write(api_code)
        with open("Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_code)
        st.success(f"已生成 {api_file} 和 Dockerfile，可直接docker build/运行！")
        st.code(api_code, language="python")
        st.code(dockerfile_code, language="docker")
