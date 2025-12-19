from tqdm.notebook import tqdm
import torch
import numpy as np

# --- add this helper once (top-level) ---
import torch.nn.functional as F

# PyTorch乱数固定用
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def soft_cross_entropy(logits, soft_targets):
    # logits: (B, C) ; soft_targets: (B, C)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()

def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history, scheduler=None):

    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs + base_epochs):
        n_train_acc, n_val_acc = 0, 0
        train_loss, val_loss = 0.0, 0.0
        n_train, n_test = 0, 0

        # 訓練フェーズ
        net.train()
        for inputs, labels in tqdm(train_loader):
            batch_size = labels.size(0)
            n_train += batch_size

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(inputs)

            # ★ MixUp/CutMix 対応: soft ラベルなら soft-CE、そうでなければ通常 CE
            if labels.ndim == 2:                      # (B, C): soft labels
                loss = soft_cross_entropy(outputs, labels)
                hard_targets = labels.argmax(dim=1)   # 精度計算用
            else:                                      # (B,): int labels
                loss = criterion(outputs, labels)
                hard_targets = labels

            loss.backward()
            optimizer.step()

            # ★ スケジューラはステップ毎に進める（SequentialLR+T_max=total_stepsなら per-iteration）
            if scheduler is not None:
                scheduler.step()

            # 予測ラベル
            predicted = outputs.argmax(dim=1)

            # 累積
            train_loss += loss.item() * batch_size
            n_train_acc += (predicted == hard_targets).sum().item()

        # 予測フェーズ（検証）に入るため、netを eval モードに
        net.eval()
        with torch.no_grad():
            for inputs_test, labels_test in test_loader:
                batch_size = labels_test.size(0)
                n_test += batch_size

                inputs_test = inputs_test.to(device, non_blocking=True)
                labels_test = labels_test.to(device, non_blocking=True)

                outputs_test = net(inputs_test)

                # ★ 検証はミキシングなし → 通常の CE
                loss_test = criterion(outputs_test, labels_test)
                predicted_test = outputs_test.argmax(dim=1)

                val_loss += loss_test.item() * batch_size
                n_val_acc += (predicted_test == labels_test).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'LR: {current_lr:.6f}')

        train_acc = n_train_acc / n_train
        val_acc   = n_val_acc / n_test
        avg_train_loss = train_loss / n_train
        avg_val_loss   = val_loss / n_test

        print(f'Epoch [{epoch+1}/{num_epochs+base_epochs}] '
              f'loss: {avg_train_loss:.5f} acc: {train_acc:.5f} '
              f'val_loss: {avg_val_loss:.5f} val_acc: {val_acc:.5f}')

        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))

    return history
