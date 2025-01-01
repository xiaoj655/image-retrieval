from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json, os
from pymilvus import MilvusClient
from torchvision.models import resnet101, ResNet101_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
from fastapi.middleware.cors import CORSMiddleware
from  fine_tune_model import model as fine_tune_model, transform

model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to('cpu')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MilvusClient(
    uri='https://in05-xxxxxxxx.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn',
    token='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATA_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

class Item(BaseModel):
    img: UploadFile
    model: str = "resnet101"

_label_dict = json.load(open('label_dict.json', 'r', encoding='u8'))
def label_dict(label: str | list[str]):
    if isinstance(label, str):
        label = label.split('#')
        return _label_dict[label[0]] + '#' + label[1]
    else:
        return list(map(label_dict, label))

from uuid import uuid4
import aiofiles


@app.post("/q")
async def post(img: UploadFile = File(...), t: int = 0, limit: int = 10):
    img_path = os.path.join(DATA_DIR, f'{uuid4()}.jpg')
    async with aiofiles.open(img_path, 'wb') as f:
        await f.write(await img.read())

    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        if t == 0:
            embedding = model(img).squeeze().tolist()
        else:
            embedding = fine_tune_model(img).squeeze().tolist()

    return client.search(
        collection_name='original_resnet101' if t == 0 else 'fine_tune_resnet101',
        data=[embedding],
        limit=limit,
        output_fields=['en_label', 'cn_label']
    )

path = os.path.join(os.path.dirname(__file__), 'movie_clip')

@app.get('/i/{movie}/{name}')
async def get(req: Request, movie: str, name: str):
    p = os.path.join(path, movie, name)
    return StreamingResponse(open(p, 'rb'), media_type='image/jpeg')


@app.get("/movie_list")
async def get():
    with open('movie_list.json', 'r', encoding='u8') as f:
        return json.load(f)