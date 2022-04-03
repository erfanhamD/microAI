import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Response, UploadFile
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse 


class Lift_base_network(nn.Module):
    def __init__(self):
        super(Lift_base_network, self).__init__()
        self.block = nn.Sequential(
        nn.Linear(5, 20),
        nn.Tanh(),
        nn.Linear(20, 8),
        nn.Tanh(),
        nn.Linear(8, 2)
        )
    def forward(self, x):
        x = self.block(x)
        return x

def preprocess(Data_addr: str):
    Data = pd.read_csv(Data_addr, header = None, delimiter=',')
    Data[4] = Data[4]/Data[0]
    scaler = pickle.load(open('x_mm_scaler.pkl', 'rb'))
    Data = scaler.transform(Data)
    return Data

def inference(Data, model):
    Data_torch = torch.from_numpy(Data)
    Cl = model(Data_torch.float())
    scaler = pickle.load(open('y_mm_scaler.pkl', 'rb'))
    Cl = Cl.cpu().detach().numpy()
    Cl_map = scaler.inverse_transform(Cl)
    return Cl_map

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def data_prep(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prompt": "Upload your data file"})

@app.post("/index", response_class=Response)
async def data_prep(request: Request, data: UploadFile = File(...)):
    if data.filename.endswith(".csv"):
        try:
            file_location = f"data/{data.filename}"
            with open(file_location, "wb+") as raw_data:
                raw_data.write(data.file.read())
            model = Lift_base_network()
            model.load_state_dict(torch.load("model_state_dict_3Apr_mm"))
            Data = preprocess(file_location)
            Cl_map = inference(Data, model)
            file_path = "data/Cl_map_"+data.filename
            download_file_name = "Cl_map_"+data.filename
            np.savetxt(file_path,Cl_map)
            return FileResponse(path = file_path, media_type="text/csv", filename = download_file_name)
        except:
            return templates.TemplateResponse("index.html", {"request": request, 'prompt': 'csv file does not match the format'})
    else:
        return templates.TemplateResponse("index.html", {"request": request, 'prompt': 'File extension not supported'})
