from src.model import VGG_16_v6 as MyCNN
from src.data_loader import data_loader
from src.trainer import trainer
#import yaml

def main():
    # 1. Load Config
    #with open("config.yaml", "r") as f:
    #    config = yaml.safe_load(f)
    # 2. Setup Data
    #train_loader, val_loader, train_dataset = get_loaders(config)
    train_loader, val_loader, train_dataset = data_loader()
    # 3. Initialize Model
    model = MyCNN(num_classes=len(train_dataset.classes))
    # 4. Train the model
    trainer(model, train_loader, val_loader, train_dataset)