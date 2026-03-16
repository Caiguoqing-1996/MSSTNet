

'''
Usage:


'''



import torch
import time
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import random
import copy


CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _split_train_val_tensor(X, Y, val_ratio=0.2):
    assert 0.0 < val_ratio < 1.0
    N = X.size(0)
    g = torch.Generator()
    # g.manual_seed(seed)
    perm = torch.randperm(N)

    n_val = int(round(N * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return (X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx])


@torch.no_grad()
def _evaluate(model, loss_func, data_loader, device, CUDA=True):

    model.eval()
    loss_func.eval()

    labels_all, preds_all, loss_list = [], [], []
    for inputs, targets in data_loader:
        if CUDA:
            inputs, targets = inputs.to(device), targets.to(device)

        results = model(inputs)
        loss_ALL, result_prob, loss_ce = loss_func(results, targets)

        pred_batch = torch.argmax(result_prob, dim=1)
        preds_all.append(pred_batch.cpu().numpy())
        labels_all.append(targets.cpu().numpy())
        loss_list.append(loss_ALL.item())

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    loss_sum = float(np.sum(np.array(loss_list)))
    acc = accuracy_score(preds_all, labels_all)
    precision = precision_score(preds_all, labels_all, average='macro', zero_division=0)
    f1 = f1_score(preds_all, labels_all, average='macro', zero_division=0)
    kappa = cohen_kappa_score(preds_all, labels_all)
    return loss_sum, acc, precision, f1, kappa



def train_in_one_fold(train_set_all, test_set, model, loss_func, train_para):
    """
    输入输出不改：
    - 输入：train_set_all/test_set: dict with 'X','Y'
    - 返回：best_acc, best_model, best_precision, best_f1, best_kappa
    """
    batch_size = train_para['batch_size']
    first_epochs = train_para.get('first_epochs', 1000)
    second_epochs = train_para.get('second_epochs', 400)
    min_train_epoch = train_para.get('min_train_epoch', 0)
    patience = train_para.get('patience', 50)
    lr = train_para.get('lr', 1e-3)
    val_ratio = train_para.get('val_ratio', 0.2)

    X_all = train_set_all['X']
    Y_all = train_set_all['Y']

    # 1) 划分 train/val（在“增强后的训练集”上做 8:2）
    (X_tr, Y_tr), (X_va, Y_va) = _split_train_val_tensor(X_all, Y_all, val_ratio=val_ratio)

    train_ds = TensorDataset(X_tr, Y_tr)
    val_ds = TensorDataset(X_va, Y_va)
    test_ds = TensorDataset(test_set['X'], test_set['Y'])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if CUDA:
        model.to(device)
        loss_func.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Stage 1: select checkpoint by lowest val loss
    # -------------------------
    best_val_loss = float('inf')
    best_val_acc = 0.00
    best_epoch_stage1 = 999
    epochs_no_improve = 0

    best_model_stage1 = None
    best_opt_stage1 = None
    stage1_train_CE_loss_at_best = None  # 用于 stage2 的 stopping threshold

    for epoch in range(first_epochs):
        epoch_in_time = time.time()

        # ---- train on train subset ----
        model.train()
        loss_func.train()
        pred_train, labels_train, loss_train_list = [], [], []
        for inputs, targets in train_loader:
            if CUDA:
                inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            results = model(inputs)
            loss_all, result_prob, loss_ce = loss_func(results, targets)
            loss_all.backward()
            optimizer.step()

            pred_labels_batch = torch.argmax(result_prob, dim=1)
            pred_train.append(pred_labels_batch.cpu().numpy())
            labels_train.append(targets.cpu().numpy())
            loss_train_list.append(loss_ce.item())

        pred_train = np.concatenate(pred_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        acc_train = accuracy_score(pred_train, labels_train)
        stage1_train_CE_loss = float(np.sum(np.array(loss_train_list)))

        # ---- val loss for checkpoint selection ----
        val_loss, val_acc, _, _, _ = _evaluate(model, loss_func, val_loader, device, CUDA=CUDA)
        # # # ---- test monitoring each epoch (only loss + acc) ----
        # test_loss, test_acc, _, _, _ = _evaluate(model, loss_func, test_loader, device, CUDA=CUDA)



        # print: only test loss/acc
        print(
            f"[Stage1][Epoch {epoch + 1:04d}] "
            f"train_CE_loss={stage1_train_CE_loss:.4f} train_acc={acc_train:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            # f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} |  "
            f"best_val_acc={best_val_acc:.4f} @Epoch {best_epoch_stage1:04d} "
            f"time={time.time() - epoch_in_time:.2f}s   "
        )

        improved = False
        if epoch >= min_train_epoch:
            # if (val_loss < best_val_loss):
            if (val_acc > best_val_acc) or ((val_acc == best_val_acc) and (val_loss < best_val_loss)):
                improved = True
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch_stage1 = epoch + 1
                stage1_train_CE_loss_at_best = stage1_train_CE_loss
                best_model_stage1 = copy.deepcopy(model.state_dict())
                best_opt_stage1 = copy.deepcopy(optimizer.state_dict())
                epochs_no_improve = 0

                print(
                    f"  >>> Stage1 checkpoint update: val_loss={best_val_loss:.4f} "
                    f"(Epoch {best_epoch_stage1:04d}), stage1_train_CE_loss_at_best={stage1_train_CE_loss_at_best:.4f}"
                )
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[Stage1] Early stop: no val_loss improvement for {patience} epochs.")
            break

    print("*" * 100)

    if best_model_stage1 is None:
        # 防御：极端情况下没保存到 checkpoint
        best_model_stage1 = copy.deepcopy(model.state_dict())
        best_opt_stage1 = copy.deepcopy(optimizer.state_dict())
        stage1_train_CE_loss_at_best = stage1_train_CE_loss_at_best if stage1_train_CE_loss_at_best is not None else float('inf')

    # -------------------------
    # Stage 2: fine-tune on full training data; final best_model selected HERE (by test_acc)
    # -------------------------
    full_train_ds = TensorDataset(X_all, Y_all)
    full_train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True)

    # resume model + optimizer from stage1 best checkpoint
    model.load_state_dict(best_model_stage1)
    if best_opt_stage1 is not None:
        optimizer.load_state_dict(best_opt_stage1)
    stop_threshold = float(stage1_train_CE_loss_at_best) if stage1_train_CE_loss_at_best is not None else float('inf')
    print(f"[Stage2] Resume from Stage1 best checkpoint. Stop when stage2_train_CE_loss < {stop_threshold:.4f}. Max epochs={second_epochs}.")
    print("Switching to fine-tuning crop ratio: [0.9, 1.0]")
    model.min_crop_ratio = 0.9
    model.max_crop_ratio = 1.0


    # ---------- Stage2 LR inspection & decay (fine-tuning) ----------
    print("[Stage2] LR before fine-tune (after resuming optimizer):")
    for gi, pg in enumerate(optimizer.param_groups):
        print(f"  - param_group[{gi}] lr = {pg.get('lr', None)}")

    # fine-tuning: set lr to half of current lr
    for pg in optimizer.param_groups:
        if 'lr' in pg and pg['lr'] is not None:
            pg['lr'] = 0.0002

    print("[Stage2] LR after halving for fine-tune:")
    for gi, pg in enumerate(optimizer.param_groups):
        print(f"  - param_group[{gi}] lr = {pg.get('lr', None)}")

    # Stage2: final best tracked by test_acc (tie -> lower test_loss)
    best_acc = -1.0
    best_test_loss = float('inf')
    best_model = None
    best_precision = 0.0
    best_f1 = 0.0
    best_kappa = 0.0

    for epoch in range(second_epochs):
        epoch_in_time = time.time()
        # ---- train on full training data ----
        model.train()
        loss_func.train()

        pred_train, labels_train, loss_train_list = [], [], []
        for inputs, targets in full_train_loader:
            if CUDA:
                inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            results = model(inputs)
            loss_all, result_prob, loss_ce = loss_func(results, targets)
            loss_all.backward()
            optimizer.step()

            pred_labels_batch = torch.argmax(result_prob, dim=1)
            pred_train.append(pred_labels_batch.cpu().numpy())
            labels_train.append(targets.cpu().numpy())
            loss_train_list.append(loss_ce.item())

        pred_train = np.concatenate(pred_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0)
        acc_train = accuracy_score(pred_train, labels_train)
        stage2_train_CE_loss = float(np.sum(np.array(loss_train_list)))

        # ---- test monitoring each epoch (only loss + acc printed) ----
        test_loss, test_acc, test_precision, test_f1, test_kappa = _evaluate(model, loss_func, test_loader, device, CUDA=CUDA)

        # print each epoch: current vs best (only loss/acc)
        # 你要求对比“当前 test_acc”和“best_model 下的 test_acc”

        print(
            f"[Stage2][Epoch {epoch+1:04d}] "
            f"train_loss={stage2_train_CE_loss:.4f} train_acc={acc_train:.4f} |   "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} |  "
            f"stop_threshold={stop_threshold:.4f} "
            f"time={time.time()-epoch_in_time:.2f}s   "
        )


        # stop rule: ONLY based on training loss threshold
        if epoch > 49:
            if stage2_train_CE_loss < stop_threshold:
                best_acc = test_acc
                best_test_loss = test_loss
                best_model = copy.deepcopy(model.state_dict())
                best_precision = test_precision
                best_f1 = test_f1
                best_kappa = test_kappa
                print(f"  >>> Stage2 BEST UPDATE: best_test_acc={best_acc:.4f} (best_test_loss={best_test_loss:.4f})")
                print(f"[Stage2] Stop condition met: stage2_train_CE_loss {stage2_train_CE_loss:.4f} < {stop_threshold:.4f}")
                break

    print("*" * 100)

    if best_model is None:
        best_model = copy.deepcopy(model.state_dict())
        # 对当前模型再算一次完整 test 指标
        test_loss, best_acc, best_precision, best_f1, best_kappa = _evaluate(model, loss_func, test_loader, device, CUDA=CUDA)

    return best_acc, best_model, best_precision, best_f1, best_kappa



if __name__ == "__main__":
    run_code = 0
