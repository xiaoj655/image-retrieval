from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from glob import glob
import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import random
import time
import logging
from pytorch_metric_learning.losses import NTXentLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='train.log'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_NUM = 60000

torch.random.manual_seed(3407)
class MovieClipDataset(Dataset):
    def __init__(
        self,
        data_dir: str, 
        gaussian_blur_prob: float=0.3, 
        random_crop_prob: float=0.4, 
        random_flip_prob: float=0.3, 
        color_jitter_prob: float = 0.6,
        sample_num: int = SAMPLE_NUM
    ):
        assert sum([gaussian_blur_prob, random_crop_prob, random_flip_prob]) == 1

        self.prob = [gaussian_blur_prob, random_crop_prob, random_flip_prob]
        for i in range(1, len(self.prob)):
            self.prob[i] = self.prob[i] + self.prob[i-1]

        self._color_jitter_prob = color_jitter_prob

        self.data = []
        for d, s, f in os.walk(data_dir):
            self.data.extend(glob(os.path.join(d, '*.jpg')))

        if sample_num>0:
            self.data = random.sample(self.data, sample_num)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self._color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.random_crop = transforms.RandomResizedCrop(224, scale=(0.75, 0.95))

    def __len__(self):
        return len(self.data)
    
    def _gaussian_blur(self, img: Image.Image):
        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 3)))
    
    def _random_crop(self, img: Image.Image):
        return self.random_crop(img)
    
    def _random_flip(self, img: Image.Image):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    def random_augment(self, img):
        _r = random.random()
        if _r < self.prob[0]:
            img = self._gaussian_blur(img)
        elif _r < self.prob[1]:
            img = self._random_crop(img)
        else:
            img = self._random_flip(img)
        _r = random.random()
        if _r < self._color_jitter_prob:
            img = self._color_jitter(img)
        return img
    
    def __getitem__(self, idx):
        anchor_img = Image.open(self.data[idx]).convert('RGB')

        positive_img = self.random_augment(anchor_img)
        positive_img = self.transform(positive_img)

        anchor_img = self.transform(anchor_img)

        # 0 原图, 1正样本
        return anchor_img, positive_img
    
    def sample(self, idx):
        anchor_img = Image.open(self.data[idx]).convert('RGB')

        positive_img = self.random_augment(anchor_img)

        return anchor_img, positive_img
        

class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()

        self.resnet = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2,
        )
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-1],
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(2048, 2048),
        )


    def forward(self, x):
        return self.resnet(x)


class SiameseNetwork(nn.Module):
    def __init__(self, base_network: nn.Module):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward(self, *args):
        return (self.base_network(x) for x in args)

# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         positive_distance = F.pairwise_distance(anchor, positive)
#         return positive_distance.mean()


def train(
    model: nn.Module,
    base_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoches: int,
    dataset: Dataset,
    save_to: str,
    batch_size: int
):
    from tqdm import tqdm

    best_loss = float(0.05)

    for epoch in tqdm(range(epoches)):
        base_network.train()
        model.train()

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=14,
            pin_memory=True,
            prefetch_factor=4
        )
        for anchor, positive in tqdm(data_loader):
            anchor, positive = anchor.to(device), positive.to(device)
            anchor_embedding, positive_embedding = model(anchor, positive)

            labels = torch.arange(len(anchor_embedding)*2)
            labels[len(anchor_embedding):] = labels[:len(anchor_embedding)]

            loss = loss_fn(
                torch.cat([anchor_embedding, positive_embedding]),
                labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time.sleep(0.1)
        print(f'Epoch {epoch+1} loss: {loss.item()}')
        logging.info(f'Epoch {epoch+1} loss: {loss.item()}')
        if loss.item() < best_loss:
            logging.info(f'Save model to {save_to}, {best_loss=}, {loss.item()=}')
            torch.save(base_network.state_dict(), save_to)
            best_loss = loss.item()
        torch.save(base_network.state_dict(), 'last_epoch_dict.pth')

if __name__ == '__main__':
    EPCOS = 15
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    DATA_DIR = '/root/autodl-tmp/processed_data'
    SAVE_TO = 'fine_tune_resnet101.pth'

    dataset = MovieClipDataset(DATA_DIR)
    base_network = Resnet101().to(device)
    if os.path.exists(SAVE_TO):
        base_network.load_state_dict(torch.load(SAVE_TO, weights_only=True))
        print(f'load fine-tuned resnet101 from disk({SAVE_TO})')

    model = SiameseNetwork(base_network).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # loss_fn = TripletLoss(margin=0.01)
    loss_fn = NTXentLoss(temperature=0.07)

    train(model, base_network, optimizer, loss_fn, EPCOS, dataset, SAVE_TO, BATCH_SIZE)
    torch.save(base_network, 'model_fine_tune_resnet101.pth')
    # os.system("/usr/bin/shutdown")
