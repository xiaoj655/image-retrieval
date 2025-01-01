import logging.config
from pymilvus import MilvusClient
import fine_tune_model
import os
from torch.utils.data import Dataset, DataLoader
import json

CLUSTER_ENDPOINT = "https://in03-e5xxxxx.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "xxxxx447a2670837757485cbexxxxxxxxxxxxxxxx1e2c5a3"

# 1. Set up a Milvus client
client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN 
)

from tqdm import tqdm
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

DATA_DIR = '/root/autodl-tmp/movie_clip'

with open('label_dict.json', 'r') as f:
    label_dict = json.load(f)

class MovieClipDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.data = []
        for d in os.listdir(data_dir):
            self.data.extend(glob(os.path.join(data_dir, d, '*.jpg')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        _name, _frame = self.data[idx].split('/')[-2:]

        en_label = _name + '#' + _frame
        cn_label = label_dict.get(_name, _name) + '#' + _frame

        return fine_tune_model.transform(img), en_label, cn_label

if __name__ == '__main__':
    from glob import glob
    from PIL import Image

    dataloader = DataLoader(
        MovieClipDataset(DATA_DIR), 
        batch_size=128, 
        num_workers=14,
    )

    for data, en_label, cn_label in tqdm(dataloader):
        data = data.to(fine_tune_model.device)
        embs = fine_tune_model.model(data).squeeze().tolist()
        _data = [
            {
                'vector': _emb,
                'en_label': _en_label,
                'cn_label': _cn_label
            } for _emb, _en_label, _cn_label in zip(embs, en_label, cn_label)
        ]
        ret = client.insert(
            collection_name='resnet101_movie_clip',
            data=_data
        )
