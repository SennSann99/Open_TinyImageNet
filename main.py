from src.model import MyCNN
from src.data_loader import get_loaders
from src.trainer import Trainer
import yaml

def main():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # 2. Setup Data
    train_loader, val_loader, train_dataset = get_loaders(config)
    
    # 3. Initialize Model
    model = MyCNN(num_classes=len(train_dataset.classes))
    trainer = Trainer(model, train_loader, val_loader, config)
    # 4. trainer.fit()
    trainer.train()