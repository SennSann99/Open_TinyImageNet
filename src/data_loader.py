from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data import default_collate
from torchvision.transforms import v2

def data_loader():
    
    transform_train = transforms.Compose([
        transforms.Resize((96, 96)),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #Normalizing pixel values ensures data is centered around zero, leading to faster and more efficient training.
        transforms.RandomErasing(p=0.5, scale = (0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ])

    train_dataset = datasets.ImageFolder(
        root="E:/datasets/tiny_image_net/tiny-imagenet-200/train",
        transform=transform_train
    )

    val_dataset = datasets.ImageFolder(
        #root="E:/datasets/tiny_image_net/tiny-imagenet-200-processed/val",
        root = "E:/datasets/tiny_image_net/tiny-imagenet-200/val_reorg",
        transform=transforms.Compose([
            transforms.Resize((96, 96)),  # Resize images to 64x64
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization
        ])
    )

    # CHeck overlap between train and val datasets
    # クラス数・名前
    train_ds = train_dataset
    val_ds = val_dataset
    print("train classes:", len(train_ds.classes))
    print("val classes:", len(val_ds.classes))

    # クラス集合と順序が同じか？
    print("same set:", set(train_ds.classes) == set(val_ds.classes))
    print("same order:", train_ds.classes == val_ds.classes)

    # ファイル名の重複チェック
    train_files = set([p.split("\\")[-1] for p, _ in train_ds.samples])
    val_files   = set([p.split("\\")[-1] for p, _ in val_ds.samples])
    print("filename overlap:", len(train_files & val_files))

    # クラスの順序の確認
    print(f"Training set: {len(train_dataset)} samples, {len(train_dataset.classes)} classes")
    print(f"Validation set: {len(val_dataset)} samples, {len(val_dataset.classes)} classes")
    print("First 10 train classes:", train_dataset.classes[:10])
    print("First 10 val classes:", val_dataset.classes[:10])


    batch_size = 100

    # Not adding CutMix and MixUp
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Add CutMix and Mixup

    NUM_CLASSES = len(train_dataset.classes)
    print(f"Number of classes: {NUM_CLASSES}")

    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    mixup = v2.MixUp(num_classes=NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


    def collate_fn(batch):
        # Implement CutMix and Mixup here if needed
        return cutmix_or_mixup(*default_collate(batch))

    # use num_workers=2 for faster data loading
    # use pin_memory=True for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    for image, label in train_loader:
        print(f"{image.shape = }, {label.shape = }")
        # No need to call cutmix_or_mixup, it's already been called as part of the DataLoader!
        break