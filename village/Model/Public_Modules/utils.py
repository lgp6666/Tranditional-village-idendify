import sys
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
import Config.pytorch as pytorch_model


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, epochs, model_name=''):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # 累计损失
    accu_loss = torch.zeros(1).to(device)
    # 累计预测正确的样本数
    accu_num = torch.zeros(1).to(device)

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
        loss.backward()
        accu_loss += loss.detach()

        optimizer.step()
        optimizer.zero_grad()

        s_loss, s_acc = pytorch_model.evaluation_metrics_al(
            accu_loss.item() / (step + 1), accu_num.item() / sample_num, model_name
        )
        data_loader.desc = f"[train epoch {epoch + 1}/{epochs}] loss: {s_loss:.3f}, acc: {s_acc:.3f}, lr: {optimizer.param_groups[0]['lr']:.5f}"

    # 每个 epoch 调用 lr_scheduler
    lr_scheduler.step()

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    loss, acc, P, R, F1 = pytorch_model.evaluation_metrics(
        accu_loss.item() / len(data_loader), accu_num.item() / sample_num, precision, recall, f1, model_name
    )
    print(f"Epoch: {epoch+1}, Train_Loss: {loss:.4f}, Train_Accuracy: {acc:.4f}, Train_Precision: {P:.4f}, Train_Recall: {R:.4f}, Train_F1_Score: {F1:.4f}")
    return loss, acc, P, R, F1


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, model_name=''):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

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

        s_loss, s_acc = pytorch_model.evaluation_metrics_al(
            accu_loss.item() / (step + 1), accu_num.item() / sample_num, model_name
        )
        data_loader.desc = f"[valid epoch {epoch+1}] loss: {s_loss:.3f}, acc: {s_acc:.3f}"

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    loss, acc, P, R, F1 = pytorch_model.evaluation_metrics(
        accu_loss.item() / len(data_loader), accu_num.item() / sample_num, precision, recall, f1, model_name
    )
    print(f"Epoch: {epoch+1}, Val_Loss: {loss:.4f}, Val_Accuracy: {acc:.4f}, Val_Precision: {P:.4f}, Val_Recall: {R:.4f}, Val_F1_Score: {F1:.4f}")
    return loss, acc, P, R, F1



def set_inplace_false(model):
    def recursive_set_inplace_false(module):
        for child_name, child in module.named_children():
            if isinstance(child, torch.nn.ReLU):
                setattr(module, child_name, torch.nn.ReLU(inplace=False))
                # print(f"Set {child_name} inplace to False")
            elif isinstance(child, torch.nn.modules.activation.ReLU):
                child.inplace = False
                # print(f"Set {child_name} inplace to False")
            else:
                recursive_set_inplace_false(child)

    recursive_set_inplace_false(model)
    return model


if __name__ == '__main__':
    pass
