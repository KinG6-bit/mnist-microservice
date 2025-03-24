from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np

app = FastAPI()

# 强制使用 CPU（Render 免费实例不支持 GPU）
device = torch.device("cpu")

# 加载模型并放到 CPU 上
model = torch.jit.load("softmax_regression_scripted.pt", map_location=device)
model.to(device)
model.eval()

# 定义输入格式
class InputData(BaseModel):
    input: list  # 3D list: [1, 28, 28]

@app.post("/predict")
async def predict(data: InputData):
    try:
        # 将输入转换为 tensor，并放到 CPU 上
        arr = np.array(data.input, dtype=np.float32)
        if arr.shape != (1, 28, 28):
            raise ValueError("Input must be of shape [1, 28, 28]")

        tensor = torch.tensor(arr).unsqueeze(0).to(device)  # [1, 1, 28, 28]

        with torch.no_grad():
            output = model(tensor)
            scores = output.squeeze().tolist()
            pred_class = int(torch.argmax(output))

        return {
            "predicted_class": pred_class,
            "scores": scores
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "MNIST classifier is up! Use POST /predict."}
