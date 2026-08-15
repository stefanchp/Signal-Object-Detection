import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# hiperparametri
ROOT_DIR = Path("..").resolve() 

TRAIN_DIR = ROOT_DIR / "train"
TEST_DIR = ROOT_DIR / "test"
TRAIN_CSV = ROOT_DIR / "train.csv"
TEST_CSV = ROOT_DIR / "test.csv"

SAVE_DIR = ROOT_DIR / "save"
MODEL_DIR = SAVE_DIR / "models"
SUBMISSION_DIR = SAVE_DIR / "submissions"
PLOTS_DIR = SAVE_DIR / "plots"         
WORKING_FOLD_DIR = MODEL_DIR / "kfold_working" 

for folder in [MODEL_DIR, WORKING_FOLD_DIR, SUBMISSION_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# pseudo label in stage 2 pentru sample urile cu confidence >= 0.9
PSEUDO_LABEL_THRESHOLD = 0.90

NUM_CLASSES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 110
BATCH_SIZE = 16

N_SPLITS = 5
SEED = 42

ALPHA = 0.50 # ponderea loss ului de regresie
EPOCHS_STAGE1 = 90
EPOCHS_STAGE2 = 90
EARLY_STOP_PATIENCE = 35 

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.02

USE_TTA_HFLIP = True   # facem media dintre predictia pe imaginea originala si cea flipped la inferenta

# probabilitati de augumentare exclusive 
MOSAIC4_PROB = 0.10
MOSAIC2_PROB = 0.12
SPEC_AUG_PROB = 0.20
HFLIP_PROB = 0.18
AFFINE_PROB = 0.20

# hiperparam pentru specaugment
SPEC_FREQ_MASKS = 1
SPEC_TIME_MASKS = 1
SPEC_MAX_FREQ_FRAC = 0.10
SPEC_MAX_TIME_FRAC = 0.10
SPEC_MASK_VALUE = 0.0

# augmentarea afina 
AUG_AFFINE_TRANSLATE = (0.006, 0.006)
AUG_AFFINE_SCALE = (0.97, 1.03)

# setam seedul pentru reproductibilitate
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


#  date + calcul mean/std
df = pd.read_csv(TRAIN_CSV).copy()
test_df = pd.read_csv(TEST_CSV).copy()

# retinem splitul de provenienta
df["source"] = "train"
test_df["source"] = "test"

print(df.head())
print("Total train= ", len(df))
print("Total test= ", len(test_df))


def compute_rgb_mean_std(dataframe, image_dir, image_height, image_width):
    # calculez media si deviatia standard pe fiecare canal, ca sa pot normaliza dupa
    image_dir = Path(image_dir)
    base_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
    ])

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    for image_name in tqdm(dataframe["id"].values):
        image = Image.open(image_dir / image_name).convert("RGB")
        tensor = base_transform(image)
        channel_sum += tensor.sum(dim=(1, 2))
        channel_sum_sq += (tensor ** 2).sum(dim=(1, 2))
        num_pixels += tensor.shape[1] * tensor.shape[2]

    mean = channel_sum / num_pixels
    # var = E[x^2] - E[x]^2, apoi sqrt ca sa am std
    std = (channel_sum_sq / num_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


TRAIN_MEAN, TRAIN_STD = compute_rgb_mean_std(df, TRAIN_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
print("TRAIN_MEAN= ", TRAIN_MEAN)
print("TRAIN_STD= ", TRAIN_STD)

# transformarea pentru validare si test: fara augmentari
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
])



# creare dataseturi si augmentari
def resolve_image_path(row):
    if row.get("source", "train") == "test":
        return TEST_DIR / row["id"]
    return TRAIN_DIR / row["id"]


class SignalDataset(Dataset):
    # fara augmentari, pentru validare si test
    def __init__(self, dataframe, transform, has_labels):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(resolve_image_path(row)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.has_labels:
            # etichetele in csv sunt [1,5] le trecem in [0, 4]
            label = int(row["label"]) - 1
            return image, torch.tensor(label, dtype=torch.long)
        return image, row["id"]


class SpecAugmentTensor:
    # SpecAugment = pun benzi la intamplare pe verticala (frecventa) si orizontala (timp)
    def __init__(self, freq_masks, time_masks, max_freq_frac,
                 max_time_frac, mask_value):
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.max_freq_frac = max_freq_frac
        self.max_time_frac = max_time_frac
        self.mask_value = mask_value

    def __call__(self, x):
        _, h, w = x.shape
        # masti pe axa de frecventa
        for _ in range(self.freq_masks):
            max_f = max(1, int(h * self.max_freq_frac))
            f = random.randint(1, max_f)
            f0 = random.randint(0, max(0, h - f))
            x[:, f0:f0 + f, :] = self.mask_value
        # masti pe axa de timp
        for _ in range(self.time_masks):
            max_t = max(1, int(w * self.max_time_frac))
            t = random.randint(1, max_t)
            t0 = random.randint(0, max(0, w - t))
            x[:, :, t0:t0 + t] = self.mask_value
        return x


class ExclusiveAugSignalDataset(Dataset):
    # datasetul de train, aplicam maxim o augumentare pe imagine
    def __init__(self, dataframe, mean, std, image_height, image_width, num_classes,
                 mosaic4_prob, mosaic2_prob, spec_aug_prob, hflip_prob, affine_prob):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.mean = mean
        self.std = std
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes

        self.mosaic4_prob = mosaic4_prob
        self.mosaic2_prob = mosaic2_prob
        self.spec_aug_prob = spec_aug_prob
        self.hflip_prob = hflip_prob
        self.affine_prob = affine_prob

        self.labels_1based = self.dataframe["label"].astype(int).values
        self.resize = transforms.Resize((image_height, image_width))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        # umplem colturile imaginii cu media pe train denormalizata
        self.affine_fill = [float(m * 255.0) for m in mean]
        self.specaug = SpecAugmentTensor(
            freq_masks=SPEC_FREQ_MASKS,
            time_masks=SPEC_TIME_MASKS,
            max_freq_frac=SPEC_MAX_FREQ_FRAC,
            max_time_frac=SPEC_MAX_TIME_FRAC,
            mask_value=SPEC_MASK_VALUE,
        )

    def __len__(self):
        return len(self.dataframe)

    def _load_pil(self, idx):
        row = self.dataframe.iloc[idx]
        return Image.open(resolve_image_path(row)).convert("RGB")

    def _label_1based(self, idx):
        return int(self.labels_1based[idx])

    def _base_tensor(self, image):
        image = self.resize(image)
        return self.normalize(self.to_tensor(image))

    def _choose_aug(self):
       # alegem augumentarea random
        r = random.random()
        if r < self.mosaic4_prob:
            return "mosaic4"
        r -= self.mosaic4_prob
        if r < self.mosaic2_prob:
            return "mosaic2"
        r -= self.mosaic2_prob
        if r < self.spec_aug_prob:
            return "specaug"
        r -= self.spec_aug_prob
        if r < self.hflip_prob:
            return "hflip"
        r -= self.hflip_prob
        if r < self.affine_prob:
            return "affine"
        return "none"

    def _sample_indices_for_mosaic(self, base_label, extra_count):
        # pentru mosaic eticheta noua e suma etichetelor
        # selectez imagini ai suma sa nu depaseasca clasa maxima
        for _ in range(80):
            indices = np.random.randint(0, len(self.dataframe), size=extra_count).tolist()
            total = base_label + sum(self._label_1based(i) for i in indices)
            if total <= self.num_classes:
                return indices, total
        return None, base_label

    def _make_mosaic2(self, idx):
        # lipesc 2 imagini una langa alta 
        base_label = self._label_1based(idx)
        indices, total_label = self._sample_indices_for_mosaic(base_label, 1)
        if indices is None:
            # n-am gasit nicio pereche, returnez imaginea originala
            return self._base_tensor(self._load_pil(idx)), base_label - 1

        imgs = [self.resize(self._load_pil(idx)), self.resize(self._load_pil(indices[0]))]
        w, h = self.image_width, self.image_height
        canvas = Image.new("RGB", (w, h), tuple(int(m * 255) for m in self.mean))
        canvas.paste(imgs[0].resize((w // 2, h)), (0, 0))
        canvas.paste(imgs[1].resize((w - w // 2, h)), (w // 2, 0))
        return self.normalize(self.to_tensor(canvas)), total_label - 1

    def _make_mosaic4(self, idx):
        # mosaic cu 4 imagini
        base_label = self._label_1based(idx)
        indices, total_label = self._sample_indices_for_mosaic(base_label, 3)
        if indices is None:
            return self._base_tensor(self._load_pil(idx)), base_label - 1

        all_indices = [idx] + indices
        imgs = [self.resize(self._load_pil(i)) for i in all_indices]
        w, h = self.image_width, self.image_height
        w1, h1 = w // 2, h // 2
        canvas = Image.new("RGB", (w, h), tuple(int(m * 255) for m in self.mean))
        positions = [(0, 0), (w1, 0), (0, h1), (w1, h1)]
        sizes = [(w1, h1), (w - w1, h1), (w1, h - h1), (w - w1, h - h1)]
        for img, pos, size in zip(imgs, positions, sizes):
            canvas.paste(img.resize(size), pos)
        return self.normalize(self.to_tensor(canvas)), total_label - 1

    def __getitem__(self, idx):
        aug = self._choose_aug()

        # pt mosaic eticheta = suma etichetelor img componente
        if aug == "mosaic4":
            x, label0 = self._make_mosaic4(idx)
            return x, torch.tensor(label0, dtype=torch.long)

        if aug == "mosaic2":
            x, label0 = self._make_mosaic2(idx)
            return x, torch.tensor(label0, dtype=torch.long)

        # la restul augumentarilor eticheta originala
        image = self.resize(self._load_pil(idx))
        label0 = self._label_1based(idx) - 1

        if aug == "hflip":
            image = TF.hflip(image)
        elif aug == "affine":
            # translatie + scale
            max_dx = int(round(AUG_AFFINE_TRANSLATE[0] * self.image_width))
            max_dy = int(round(AUG_AFFINE_TRANSLATE[1] * self.image_height))
            translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
            scale = random.uniform(AUG_AFFINE_SCALE[0], AUG_AFFINE_SCALE[1])
            image = TF.affine(image, angle=0.0, translate=translate, scale=scale,
                              shear=[0.0, 0.0], fill=self.affine_fill)

        x = self.normalize(self.to_tensor(image))
        if aug == "specaug":
            # specaug aplicat pe tensor dupa normalizare
            x = self.specaug(x)
        return x, torch.tensor(label0, dtype=torch.long)


# arhitectura CNN

 # squeeze and excitation block: invata cat de important e fiecare canal si il pondereaza
class SEBlock(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)   # comprim fiecare canal la o singura valoare
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),   # ponderi intre 0 si 1 pentru fiecare canal
        )
    # trecem inputul prin bloc, prin layerele de squeeze si excite
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w   # reponderez canalele


class ResidualConvBlock(nn.Module):
    # bloc rezidual cu 2 convolutii + skip connection
    def __init__(self, in_channels, out_channels, dropout, downsample):
        super().__init__()
        stride = 2 if downsample else 1   # downsample = injumatatesc rezolutia
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential() # returneaza identitatea
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ) 
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv_block(x)
        x = self.relu(x + identity)   # adun skip connection ul inainte de relu
        return self.dropout(x)


class SignalCNN(nn.Module):
    # doua capete care pleaca din aceleasi features:
    #   cap de clasificare = logits pe 5 clase
    #   cap de regresie = o valoare in [0, 1]
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # blocuri reziduale urmate de SE blocks, pentru a selecta canalele relevante 
        # cresc numarul de filtre si scad rezolutia treptat
        # dropout creste pe masura ce mergem in adancime pentru a preveni overfit
        self.features = nn.Sequential(
            ResidualConvBlock(32, 64, dropout=0.02, downsample=True),
            SEBlock(64, reduction=16),
            ResidualConvBlock(64, 64, dropout=0.02, downsample=False),
            SEBlock(64, reduction=16),
            ResidualConvBlock(64, 128, dropout=0.04, downsample=True),
            SEBlock(128, reduction=16),
            ResidualConvBlock(128, 128, dropout=0.04, downsample=False),
            SEBlock(128, reduction=16),
            ResidualConvBlock(128, 256, dropout=0.08, downsample=True),
            SEBlock(256, reduction=16),
            ResidualConvBlock(256, 256, dropout=0.08, downsample=False),
            SEBlock(256, reduction=16),
            ResidualConvBlock(256, 512, dropout=0.12, downsample=True),
            SEBlock(512, reduction=16),
            ResidualConvBlock(512, 512, dropout=0.12, downsample=False),
            SEBlock(512, reduction=16),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # global average pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
            nn.Sigmoid(),   # [0, 1]
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        reg_pred = self.regressor(x).squeeze(1)
        return logits, reg_pred



# loss combinat 
def labels_to_regression_targets(labels):
    # duc etichetele 0..4 in intervalul [0, 1] pentru capul de regresie
    return labels.float() / (NUM_CLASSES - 1)


def combined_loss(logits, reg_pred, labels, criterion_cls, criterion_reg, alpha):
    # loss ul final = combinatie intre clasificare si regresie, controlata de alpha
    cls_loss = criterion_cls(logits, labels)
    reg_loss = criterion_reg(reg_pred, labels_to_regression_targets(labels))
    loss = (1.0 - alpha) * cls_loss + alpha * reg_loss
    return loss


def make_loaders(train_dataframe, val_dataframe, batch_size=BATCH_SIZE):
    # trainul foloseste datasetul cu augmentari, validarea pe cel simplu
    train_dataset = ExclusiveAugSignalDataset(
        dataframe=train_dataframe,
        mean=TRAIN_MEAN,
        std=TRAIN_STD,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        num_classes=NUM_CLASSES,
        mosaic4_prob=MOSAIC4_PROB,
        mosaic2_prob=MOSAIC2_PROB,
        spec_aug_prob=SPEC_AUG_PROB,
        hflip_prob=HFLIP_PROB,
        affine_prob=AFFINE_PROB,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataframe is not None:
        val_dataset = SignalDataset(val_dataframe, transform=val_transform, has_labels=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    return train_loader, val_loader


def train_one_epoch(model, dataloader, criterion_cls, criterion_reg, optimizer, alpha, epoch_desc):
    model.train()
    total_loss = 0.0
    total_correct = total_seen = 0

    pbar = tqdm(dataloader, desc=epoch_desc, leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, reg_pred = model(images)   # forward pass
        loss = combined_loss(logits, reg_pred, labels, criterion_cls, criterion_reg, alpha)
        loss.backward()    # backprop
        optimizer.step()   # update parametri

        # tin sume ponderate cu batch size ca sa am medii corecte la final
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += bs
        pbar.set_postfix(loss=total_loss / total_seen, acc=total_correct / total_seen)

    return {
        "loss": total_loss / total_seen,
        "acc": total_correct / total_seen,
    }


@torch.no_grad() # evaluam modelul, disable la gradienti
def evaluate_model(model, dataloader, criterion_cls, criterion_reg, alpha):
    model.eval()
    total_loss =  0.0
    total_correct = total_seen = 0

    for images, labels in tqdm(dataloader, desc="Valid", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits, reg_pred = model(images)

        loss = combined_loss(logits, reg_pred, labels, criterion_cls, criterion_reg, alpha)
        total_loss += loss.item() * labels.size(0)

        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)

    out = {"acc": total_correct / total_seen}
    out.update({
        "loss": total_loss / total_seen,
    })
    return out


@torch.no_grad()
def collect_outputs(model, dataloader, use_tta_hflip):
    # ruleaza modelul pe tot loaderul si calc probabilitatile
    # capul de regresie contribuie doar la loss in antrenare, e ignorat la predictie
    model.eval()
    all_probs = []

    for batch in tqdm(dataloader, desc="Collect outputs", leave=False):
        images = batch[0]
        images = images.to(device)
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=1)

        if use_tta_hflip:
            # rulez si imaginea oglindita si fac media celor doua predictii
            flipped = torch.flip(images, dims=[3])
            logits_f, _ = model(flipped)
            probs_f = torch.softmax(logits_f, dim=1)
            probs = 0.5 * (probs + probs_f)

        all_probs.append(probs.cpu().numpy())

    return {
        "probs": np.concatenate(all_probs, axis=0),
    }


def train_fold(stage_name, fold, train_dataframe, val_dataframe, epochs, fold_model_path):
    # antrenez un fold de la zero si salvez checkpoint ul cu cea mai buna acuratete pe validare
    set_seed(SEED + 1000 * (1 if stage_name == "stage1" else 2) + fold) # seed diferit per stage si per fold

    train_loader, val_loader = make_loaders(train_dataframe, val_dataframe)
    model = SignalCNN(num_classes=NUM_CLASSES).to(device)

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_acc = -1.0
    best_epoch = 0
    patience = 0   
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n[{stage_name}] Fold {fold}/{N_SPLITS} | Epoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, ALPHA,
            epoch_desc=f"{stage_name} f{fold} train",
        )
        val_metrics = evaluate_model(model, val_loader, criterion_cls, criterion_reg, ALPHA)

        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"train_acc={train_metrics['acc']:.4f} | val_acc={val_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        # daca s a imbunatatit val_acc, salvez modelul si resetez patience
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            patience = 0
            # salvam weights + accuracy
            checkpoint = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "val_acc": best_acc,
            }
            torch.save(checkpoint, fold_model_path)
            print(f"Save best fold checkpoint= {fold_model_path}")
        else:
            patience += 1

        # early stopping
        if patience >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}\n Best epoch={best_epoch}, best val_acc={best_acc:.5f}")
            break

    # reincarc cel mai bun checkpoint
    checkpoint = torch.load(fold_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, best_acc, pd.DataFrame(history)

# functii de plotare 
def _save_show(fig, filename):
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Plot saved= {path}")
    plt.show()
    plt.close(fig)


def average_histories(histories):
    # fac media metricilor pe cele 5 folduri
    # plotez istoricul comun
    min_len = min(len(h) for h in histories)
    cols = ["train_acc", "val_acc", "train_loss", "val_loss"]
    stacked = {c: np.stack([h[c].values[:min_len] for h in histories], axis=0) for c in cols}
    mean = {c: stacked[c].mean(axis=0) for c in cols}
    epochs = np.arange(1, min_len + 1)
    return epochs, mean


def plot_acc_curves(histories, stage_name):
    # plotez evolutia pentru train_acc si val_acc
    epochs, mean = average_histories(histories)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean["train_acc"], "b-o", label="Acuratete antrenare")
    plt.plot(epochs, mean["val_acc"], "g-o", label="Acuratete validare")
    plt.title(f"Evolutia acuratetii pe epoci ({stage_name}, media pe {len(histories)} folduri)")
    plt.xlabel("Epoca")
    plt.ylabel("Acuratete")
    plt.grid(True)
    plt.legend()
    _save_show(fig, f"acc_curves_{stage_name}.png")


def plot_loss_curves(histories, stage_name):
    # plotez evolutia pentru train_loss si val_loss
    epochs, mean = average_histories(histories)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean["train_loss"], "b-o", label="Loss antrenare")
    plt.plot(epochs, mean["val_loss"], "g-o", label="Loss validare")
    plt.title(f"Evolutia loss-ului pe epoci ({stage_name}, media pe {len(histories)} folduri)")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    _save_show(fig, f"loss_curves_{stage_name}.png")


def plot_confusion_matrix(cm, title, filename):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.xlabel("Etichete prezise")
    plt.ylabel("Etichete reale")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks + 1)   # afisez etichetele 1..5
    plt.yticks(ticks, ticks + 1)
    th = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > th else "black")
    _save_show(fig, filename)


def plot_blend_search(blend_df, best_w):
    # variatia acuratetii OOF in functie de ponderea de blend
    fig = plt.figure(figsize=(8, 5))
    plt.plot(blend_df["w_stage2"], blend_df["oof_acc"], "b-")
    plt.axvline(best_w, color="r", linestyle="--", label=f"best w={best_w:.2f}")
    plt.title("Cautarea ponderii de blend (w_stage2 vs OOF accuracy)")
    plt.xlabel("w_stage2  (pondere Stage 2)")
    plt.ylabel("OOF accuracy")
    plt.grid(True)
    plt.legend()
    _save_show(fig, "blend_weight_search.png")


def plot_class_accuracy(stage1_probs, stage2_probs, blend_probs, y_true, num_classes):
    # acuratete per clasa pentru stage 1, stage 2 si blend
    labels = np.arange(1, num_classes + 1)

    def per_class_acc(probs):
        preds = probs.argmax(axis=1) + 1
        cm = confusion_matrix(y_true, preds, labels=labels)
        return np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    acc1 = per_class_acc(stage1_probs)
    acc2 = per_class_acc(stage2_probs)
    accb = per_class_acc(blend_probs)

    x = np.arange(num_classes)
    width = 0.25
    fig = plt.figure(figsize=(9, 5))
    plt.bar(x - width, acc1, width, label="Stage 1")
    plt.bar(x, acc2, width, label="Stage 2")
    plt.bar(x + width, accb, width, label="Blend")
    plt.title("Acuratete per clasa: Stage 1 vs Stage 2 vs Blend")
    plt.xlabel("Clasa")
    plt.ylabel("Acuratete")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.grid(True, axis="y")
    plt.legend()
    _save_show(fig, "class_accuracy_comparison.png")


def plot_test_prediction_distribution(preds, num_classes):
    # nr de predictii pentru fiecare clasa pe test
    labels = np.arange(1, num_classes + 1)
    counts = pd.Series(preds).value_counts().reindex(labels, fill_value=0).sort_index()
    fig = plt.figure(figsize=(8, 5))
    plt.bar(labels, counts.values, color="steelblue")
    plt.title("Distributia predictiilor pe test (submisia finala)")
    plt.xlabel("Clasa prezisa")
    plt.ylabel("Numar de exemple")
    plt.xticks(labels)
    for xi, c in zip(labels, counts.values):
        plt.text(xi, c, str(int(c)), ha="center", va="bottom")
    plt.grid(True, axis="y")
    _save_show(fig, "test_prediction_distribution.png")


# incarcam datele de test
test_dataset = SignalDataset(test_df, transform=val_transform, has_labels=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# incarcam etichetele pentru datele de train
y_true_all = df["label"].astype(int).values
print("Test loader length= ", len(test_dataset))


# stage 1: train pe datele originale, splituite in 5 folduri
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# pentru fiecare fold am un split de validare = out of fold (oof)
# pentru a masura performanta
stage1_oof_probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
stage1_test_probs_sum = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)
stage1_fold_accs = []
stage1_histories = [] 

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), start=1):
    print(f"STAGE 1 FOLD {fold}/{N_SPLITS}")

    train_fold_df = df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = df.iloc[val_idx].reset_index(drop=True)

    print("Train fold size= ", len(train_fold_df), " Val fold size= ", len(val_fold_df))
    print("Val distribution: ")
    print(val_fold_df["label"].value_counts().sort_index())

    fold_model_path = WORKING_FOLD_DIR / f"stage1_fold{fold}.pt"
    model, best_acc, history_df = train_fold("stage1", fold, train_fold_df, val_fold_df, EPOCHS_STAGE1, fold_model_path)
    stage1_fold_accs.append(best_acc)
    stage1_histories.append(history_df)

    # predictiile oof pe foldul de validare
    val_loader = DataLoader(
        SignalDataset(val_fold_df, transform=val_transform, has_labels=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_outputs = collect_outputs(model, val_loader, use_tta_hflip=USE_TTA_HFLIP)
    stage1_oof_probs[val_idx] = val_outputs["probs"]

    # predictiile pe test, adunate pt a putea face media peste cele 5 folduri
    test_outputs = collect_outputs(model, test_loader, use_tta_hflip=USE_TTA_HFLIP)
    stage1_test_probs_sum += test_outputs["probs"]

stage1_test_probs = stage1_test_probs_sum / N_SPLITS
stage1_oof_preds = stage1_oof_probs.argmax(axis=1) + 1   # +1 ca sa revin la 1..5
stage1_cv_acc = accuracy_score(y_true_all, stage1_oof_preds) # calc acuratetea

print("STAGE 1 DONE")
print("Fold accs= ", [round(x, 5) for x in stage1_fold_accs])
print(f"Stage 1 OOF CV accuracy= {stage1_cv_acc:.5f}")
print("Stage 1 OOF confusion matrix: ")
print(confusion_matrix(y_true_all, stage1_oof_preds))
print(classification_report(y_true_all, stage1_oof_preds, digits=4, zero_division=0))

# grafice stage 1
plot_acc_curves(stage1_histories, "stage1")
plot_loss_curves(stage1_histories, "stage1")
plot_confusion_matrix(
    confusion_matrix(y_true_all, stage1_oof_preds, labels=np.arange(1, NUM_CLASSES + 1)),
    "Matrice de confuzie - Stage 1 (OOF)",
    "confusion_matrix_stage1.png",
)

# pseudo labeling pentru predictiile din test foarte sigure
stage1_test_max_prob = stage1_test_probs.max(axis=1)       # cat de sigur e modelul pe fiecare sample
stage1_test_pred_label = stage1_test_probs.argmax(axis=1) + 1

# pastrez doar predictiile cu incredere peste threshold si le tratez ca date noi de train
pseudo_mask = stage1_test_max_prob >= PSEUDO_LABEL_THRESHOLD
pseudo_df = test_df.loc[pseudo_mask, ["id", "source"]].copy()
pseudo_df["label"] = stage1_test_pred_label[pseudo_mask].astype(int)
pseudo_df["pseudo_confidence"] = stage1_test_max_prob[pseudo_mask]

print(f"Pseudo-label threshold= {PSEUDO_LABEL_THRESHOLD}")
print(f"Pseudo-labels selected= {len(pseudo_df)} / {len(test_df)}")
print("Pseudo-label distribution: ")
print(pseudo_df["label"].value_counts().reindex(range(1, NUM_CLASSES + 1), fill_value=0).sort_index())
print("Confidence summary: ")
print(pd.Series(stage1_test_max_prob).describe())


# stage 2: datele initiale de train + cele pseudo labeled
stage2_oof_probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
stage2_test_probs_sum = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)
stage2_fold_accs = []
stage2_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), start=1):
    print(f"STAGE 2  FOLD {fold}/{N_SPLITS}")

    train_original_fold_df = df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = df.iloc[val_idx].reset_index(drop=True)

    # concatenam sample urile pseudo labeled la datele initiale de train
    if len(pseudo_df) > 0:
        train_fold_df = pd.concat([
            train_original_fold_df,
            pseudo_df[["id", "label", "source", "pseudo_confidence"]],
        ], ignore_index=True)
    else:
        train_fold_df = train_original_fold_df.copy()

    print("Original train fold size= ", len(train_original_fold_df))
    print("Pseudo labels added= ", len(pseudo_df))
    print("Total stage2 train fold size= ", len(train_fold_df))
    print("Val fold size= ", len(val_fold_df))
    print("Stage2 train distribution")
    print(train_fold_df["label"].value_counts().reindex(range(1, NUM_CLASSES + 1), fill_value=0).sort_index())

    fold_model_path = WORKING_FOLD_DIR / f"stage2_fold{fold}.pt"
    model, best_acc, history_df = train_fold("stage2", fold, train_fold_df, val_fold_df, EPOCHS_STAGE2, fold_model_path)
    stage2_fold_accs.append(best_acc)
    stage2_histories.append(history_df)

    # validare doar pe samples din train
    val_loader = DataLoader(
        SignalDataset(val_fold_df, transform=val_transform, has_labels=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_outputs = collect_outputs(model, val_loader, use_tta_hflip=USE_TTA_HFLIP)
    stage2_oof_probs[val_idx] = val_outputs["probs"]

    test_outputs = collect_outputs(model, test_loader, use_tta_hflip=USE_TTA_HFLIP)
    stage2_test_probs_sum += test_outputs["probs"]

stage2_test_probs = stage2_test_probs_sum / N_SPLITS
stage2_oof_preds = stage2_oof_probs.argmax(axis=1) + 1
stage2_cv_acc = accuracy_score(y_true_all, stage2_oof_preds)

print("STAGE 2 DONE")
print("Fold accs= ", [round(x, 5) for x in stage2_fold_accs])
print(f"Stage 2 OOF CV accuracy= {stage2_cv_acc:.5f}")
print("Stage 2 OOF confusion matrix:")
print(confusion_matrix(y_true_all, stage2_oof_preds))
print(classification_report(y_true_all, stage2_oof_preds, digits=4, zero_division=0))
print("Predicted distribution OOF:")
print(pd.Series(stage2_oof_preds).value_counts().reindex(range(1, NUM_CLASSES + 1), fill_value=0).sort_index())

# grafice stage 2
plot_acc_curves(stage2_histories, "stage2")
plot_loss_curves(stage2_histories, "stage2")
plot_confusion_matrix(
    confusion_matrix(y_true_all, stage2_oof_preds, labels=np.arange(1, NUM_CLASSES + 1)),
    "Matrice de confuzie - Stage 2 (OOF)",
    "confusion_matrix_stage2.png",
)


# blend stage 1 + stage 2
# probs = w * probs_stage2 + (1 - w) * probs_stage1

y_true = df["label"].astype(int).values
labels = np.arange(1, NUM_CLASSES + 1)

# incerc toate ponderile de la 0 la 1 din 0.01 in 0.01
weights = np.round(np.arange(0.00, 1.0001, 0.01), 2)
rows = []

for w in weights:
    blend_oof_probs = w * stage2_oof_probs + (1.0 - w) * stage1_oof_probs
    blend_oof_preds = blend_oof_probs.argmax(axis=1) + 1

    acc = accuracy_score(y_true, blend_oof_preds)
    cm = confusion_matrix(y_true, blend_oof_preds, labels=labels)
    class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)   # acuratete pe fiecare clasa

    row = {"w_stage2": w, "w_stage1": round(1.0 - w, 2), "oof_acc": acc}
    for cls, cls_acc in zip(labels, class_acc):
        row[f"class_{cls}_acc"] = cls_acc
    rows.append(row)

blend_search_df = pd.DataFrame(rows)

# aleg ponderea cu cea mai buna acuratete pe OOF
best_idx = blend_search_df["oof_acc"].idxmax()
best_row = blend_search_df.loc[best_idx]
BEST_BLEND_W = float(best_row["w_stage2"])

print("BEST BLEND FOUND ON OOF")
print(f"Best weight Stage 2= {BEST_BLEND_W:.2f}")
print(f"Best weight Stage 1= {1.0 - BEST_BLEND_W:.2f}")
print(f"Best OOF accuracy=   {best_row['oof_acc']:.5f}")

stage1_acc = accuracy_score(y_true, stage1_oof_probs.argmax(axis=1) + 1)
stage2_acc = accuracy_score(y_true, stage2_oof_probs.argmax(axis=1) + 1)

print()
print(f"Stage 1 OOF accuracy= {stage1_acc:.5f}")
print(f"Stage 2 OOF accuracy= {stage2_acc:.5f}")
print(f"Blend  OOF accuracy=  {best_row['oof_acc']:.5f}")

# evolutia acc oof in functie de weightul folosit la blend
plot_blend_search(blend_search_df, BEST_BLEND_W)

# metrici pentru cel mai bun blend
best_blend_oof_probs = BEST_BLEND_W * stage2_oof_probs + (1.0 - BEST_BLEND_W) * stage1_oof_probs
best_blend_oof_preds = best_blend_oof_probs.argmax(axis=1) + 1

cm = confusion_matrix(y_true, best_blend_oof_preds, labels=labels)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{x}" for x in labels],
    columns=[f"pred_{x}" for x in labels],
)
print("BEST BLEND - CONFUSION MATRIX OOF")
print(cm_df)

class_acc_df = pd.DataFrame({
    "class": labels,
    "correct": np.diag(cm),
    "total": cm.sum(axis=1),
    "class_acc": np.diag(cm) / np.maximum(cm.sum(axis=1), 1),
})
print("\nBEST BLEND - CLASS ACCURACY")
print(class_acc_df.round(5))

plot_confusion_matrix(
    cm,
    f"Matrice de confuzie - Blend w={BEST_BLEND_W:.2f} (OOF)",
    f"confusion_matrix_blend_w{BEST_BLEND_W:.2f}".replace(".", "p") + ".png",
)

# acuratete per clasa pentru stage 1 vs stage 2 vs blend
plot_class_accuracy(stage1_oof_probs, stage2_oof_probs, best_blend_oof_probs, y_true, NUM_CLASSES)

# blend pe probabilitatile de test pentru predictiile finale
best_blend_test_probs = BEST_BLEND_W * stage2_test_probs + (1.0 - BEST_BLEND_W) * stage1_test_probs
best_blend_test_preds = best_blend_test_probs.argmax(axis=1) + 1

submission_blend_df = pd.DataFrame({
    "id": test_df["id"].values,
    "label": best_blend_test_preds.astype(int),
})

print("BLENDED TEST PREDICTION DISTRIBUTION")
print(
    submission_blend_df["label"]
    .value_counts()
    .sort_index()
    .rename_axis("label")
    .reset_index(name="count")
)

# distributia predictiilor pe test in submisia finala
plot_test_prediction_distribution(best_blend_test_preds, NUM_CLASSES)

# save
blend_tag = f"blend_stage1_stage2_w{BEST_BLEND_W:.2f}".replace(".", "p")

submission_path = SUBMISSION_DIR / f"submission_{blend_tag}.csv"
submission_blend_df.to_csv(submission_path, index=False)
print(f"Submission saved= {submission_path}")