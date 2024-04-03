import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler


CLASS_TO_IDX = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "joy": 4,
    "sadness": 5,
    "wonder": 6,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class EmotionsDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.targets = [x for x in np.array(data.dataset.targets)[data.indices]]
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomGrayscale(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


inf_transforms = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


def get_loader(dataset: EmotionsDataset, BATCH_SIZE=16):
    distrib = Counter(dataset.targets)
    class_weights = {x: len(dataset) / distrib[x] for x in distrib.keys()}
    sample_weights = [class_weights[x] for x in dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return DataLoader(dataset, BATCH_SIZE, sampler=sampler)


def get_f1_score(true, pred):
    return sklearn.metrics.f1_score(true, pred, average="micro")


class EarlyStopping():
    def __init__(self, patience=4):
        self.last_value = 1_000_000
        self.current_patience = 0
        self.patience = patience

    def __call__(self, new_value) -> bool:
        if new_value > self.last_value:
            self.current_patience += 1
        else:
            self.current_patience = 0
        self.last_value = new_value
        if self.current_patience > self.patience:
            return True
        return False


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    criterion,
    optimizer,
    device,
    best_model_path: str,
    scheduler=None,
    early_stopping: EarlyStopping = None
):
    model.to(device)
    train_loss = []
    val_loss = []
    best_loss = 1_000_000

    for step, epoch in enumerate(range(epochs)):
        loss_per_batch = 0.0
        train_batch_f1_score = []
        val_batch_f1_score = []

        model.train()
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)

            output = model(image)
            output_pred = output.max(1)[1].data  # get the predicted class
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            loss_per_batch += loss.item() * output.shape[0]
            train_batch_f1_score.append(get_f1_score(label.cpu(), output_pred.cpu().numpy()))

        train_loss.append(loss_per_batch / len(train_loader.dataset))

        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            loss_per_batch = 0.0
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)

                output = model(image)
                output_pred = output.max(1)[1].data
                loss = criterion(output, label)

                loss_per_batch += loss.item() * output.shape[0]
                val_batch_f1_score.append(get_f1_score(label.cpu(), output_pred.cpu().numpy()))

            val_loss.append(loss_per_batch / len(val_loader.dataset))

        print(f"EPOCH: {step + 1}, train_loss: {train_loss[step]}, val_loss: {val_loss[step]}")
        print(f"train_f1: {np.mean(train_batch_f1_score)}, val_f1: {np.mean(val_batch_f1_score)}")
        print()

        if val_loss[step] < best_loss:
            best_loss = val_loss[step]
            torch.save(model.state_dict(), best_model_path)

        if early_stopping is not None:
            need_to_stop = early_stopping(val_loss[step])
            if need_to_stop is True:
                print("Early Stopping!")
                return train_loss, val_loss

    return train_loss, val_loss


def test_model(model, loader, device):
    model.eval()
    test_f1_score = []
    for image, label in loader:
        output = model(image.to(device))
        output_pred = output.max(1)[1].data
        test_f1_score.append(get_f1_score(label.cpu(), output_pred.cpu().numpy()))

    print(np.mean(test_f1_score))


# source: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = sklearn.metrics.roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

def get_metrics_report(true, pred):
    return {
        "Accuracy": sklearn.metrics.accuracy_score(true, pred),
        "Precision_macro": sklearn.metrics.precision_score(true, pred, average='macro'),
        "Precision_micro": sklearn.metrics.precision_score(true, pred, average='micro'),
        "Recall_macro": sklearn.metrics.recall_score(true, pred, average='macro'),
        "Recall_micro": sklearn.metrics.recall_score(true, pred, average='micro'),
        "ROC_AUC": roc_auc_score_multiclass(true, pred)
    }


def get_classification_report(true, pred, idx_to_class):
    t, p = np.array(true), np.array(pred)
    for _class in set(t):
        indices_class = np.where(t == _class)[0]
        correct = t[indices_class]
        predicted = p[indices_class]

        print(f"{idx_to_class[_class]} emotion")
        print(f"Overall images: {len(indices_class)}")
        print(f"Correctly predicted {(correct == predicted).sum()}/{len(indices_class)}")
        print()


def get_mistaken_images_report(path_list, actual, pred, model_name, idx_to_class, SAVE_LOGS_PATH):
    for ind, (actual, pred) in enumerate(zip(actual, pred)):
        if actual == pred:
            continue

        image = Image.open(path_list[ind]).convert("RGB")
        save_path = f'{SAVE_LOGS_PATH}/{model_name}/'
        Path(save_path).mkdir(parents=True, exist_ok=True)

        plt.title(f"True: {idx_to_class[actual]}, Predicted: {idx_to_class[pred]}")
        plt.imshow(image)
        plt.savefig(f'{save_path}image_{ind}.png')
        plt.show()


def inference_model(model, path_list, device):
    model.to(device)
    model.eval()

    pred = []
    for f in path_list:
        image = Image.open(f).convert('RGB')
        tensor = inf_transforms(image).to(device)
        predicted = model(tensor.unsqueeze(0)).max(1)[1].data.cpu().item()
        pred.append(predicted)

    return pred
