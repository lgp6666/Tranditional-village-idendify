import sys
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, epochs):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    all_preds = []
    all_labels = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        # 确保 pred 是一个 Tensor
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}/{}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch + 1,
            epochs,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(
        f'Epoch: {epoch}, Train_Loss: {accu_loss.item() / (step + 1):.4f}, '
        f'Train_Accuracy: {accu_num.item() / sample_num:.4f}, '
        f'Train_Precision: {precision:.4f}, '
        f'Train_Recall: {recall:.4f}, '
        f'Train_F1_Score: {f1:.4f}')

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, f1


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    all_preds = []
    all_labels = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Epoch: {epoch}, Val_Loss: {accu_loss.item() / (step + 1):.4f}, '
          f'Val_Accuracy: {accu_num.item() / sample_num:.4f}, '
          f'Val_Precision: {precision:.4f}, '
          f'Val_Recall: {recall:.4f}, Val_F1_Score: {f1:.4f}')

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, f1


if __name__ == '__main__':
    pass
