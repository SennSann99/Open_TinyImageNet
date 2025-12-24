import torch

import torch.optim as optim
import torch.nn as nn
import numpy as np


def trainer(model, train_loader, val_loader, train_dataset):
    from src.utils import torch_seed, fit
    
    # デバイスの割り当て
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 乱数初期化
    torch_seed()

    # モデルをデバイスに移動
    net = model.to(device)

    # 損失関数： 交差エントロピー関数
    criterion = nn.CrossEntropyLoss()

    # 学習率を0.01に設定
    lr = 0.0005

    # 最適化関数にAdamを指定
    optimizer = optim.Adam(net.parameters(), lr=lr)

    history2 = np.zeros((0, 5))

    num_epochs = 100
    #
    # -----------------------------------------------------------------------
    # 既存: optimizer などを作ったあとに追加
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # warmupの設定 10%のステップ数をウォームアップに使用
    warmup_ratio = 0.1
    warmup_steps = int(total_steps * warmup_ratio)

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    # 0.00001から始めて、 0.01 まで線形に増加させるウォームアップスケジューラ
    sched_warmup = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    # 1.0から始めて、残りのステップで 0.01 → 0.0（コサイン）
    # T_max: how many scheduler steps it takes to go from the start LR down to the minimum, following a cosine curve.
    sched_cos = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=0.0
    )
    # -----------------------------------------------------------------------
    # SequentialLRを使用して、ウォームアップとコサイン減衰を組み合わせる
    scheduler = SequentialLR(
        optimizer, schedulers=[sched_warmup, sched_cos], milestones=[warmup_steps]
    )

    # fit に scheduler を渡すようにする
    history2 = fit(net, optimizer, criterion, num_epochs,
               train_loader, val_loader, device, history2,
               scheduler=scheduler)