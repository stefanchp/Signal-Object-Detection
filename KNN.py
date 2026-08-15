import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# resize la 64x64 ca sa fie mai rapid
img_size = (64, 64)
vecini = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 61, 81, 101]

def read_images(folder, image_files):
    features_vectors = []
    for file_name in tqdm(image_files, desc="citesc " + folder):
        img = Image.open(folder + "/" + str(file_name)).convert("L")  # le fac grayscale
        img = img.resize(img_size)
        hist = np.asarray(img, dtype=np.float32) / 255.0   # impart la 255 ca sa fie intre 0 si 1
        hist = hist.reshape(-1)   # flatten
        features_vectors.append(hist)
    return np.array(features_vectors, dtype=np.float32)


def read_labels(df):
    labels = df["label"].astype(int).to_list()
    return np.array(labels)


# date
train_df = pd.read_csv("../train.csv")
test_df = pd.read_csv("../test.csv")

train_images = read_images("../train", train_df["id"].to_list())
test_images = read_images("../test", test_df["id"].to_list())
train_labels = read_labels(train_df)

# train + validare, split + stratify, pastrez proportiile 
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print("train:", X_train.shape)
print("validare:", X_val.shape)


# incercam combinatiile de metrica, weights si k
rezultate = []
curbe = {}
best_acc = -1
best_metric = None
best_weight = None
best_k = None
best_pred = None

for metric in ["euclidean", "cosine"]:
    print("\nmetrica: ", metric)
    
    if metric == "euclidean":
        scaler = StandardScaler() #standardizez
    else:
        scaler = Normalizer(norm="l2") #normalizare L2

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    for weight in ["uniform", "distance"]:
        acc = []
        for k in vecini:
            classifier = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
            classifier.fit(X_train_scaled, y_train)
            predicted_labels = classifier.predict(X_val_scaled)
            a = accuracy_score(y_val, predicted_labels)

            rezultate.append([metric, weight, k, a])
            acc.append(a)
            print("k =", k, "| weights =", weight, "| acc =", round(a, 4))

            # best config pana acum
            if a > best_acc:
                best_acc = a
                best_metric = metric
                best_weight = weight
                best_k = k
                best_pred = predicted_labels

        curbe[(metric, weight)] = acc


# afisam primele 10 configuratii descrescator dupa acuratete
rezultate_df = pd.DataFrame(rezultate, columns=["metric", "weights", "k", "acc"])
rezultate_df = rezultate_df.sort_values("acc", ascending=False)
print("\ntop 10 rezultate:")
print(rezultate_df.head(10))
print("\ncea mai buna varianta: metrica =", best_metric, ", weights =", best_weight, ", k =", best_k, ", acc =", best_acc)


# plotam acuratetea dupa k
plt.figure(figsize=(8, 5))
for metric, weight in curbe:
    plt.plot(vecini, curbe[(metric, weight)], marker="o", label=metric + " / " + weight)
plt.title("Acuratete KNN in functie de k")
plt.xlabel("Numar de vecini (k)")
plt.ylabel("Acuratete")
plt.xticks(vecini)
plt.grid(True)
plt.legend()
plt.show()


# matricea de confuzie
clase = sorted(np.unique(train_labels))
cm = confusion_matrix(y_val, best_pred, labels=clase)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clase)
disp.plot(cmap="Blues", values_format="d")
plt.title("Matricea de confuzie")
plt.xlabel("Etichete prezise")
plt.ylabel("Etichete reale")
plt.show()


# retrain pe cei mai buni parametrii + predictii
print("\nretrain for best config")
if best_metric == "euclidean":
    scaler = StandardScaler()
else:
    scaler = Normalizer(norm="l2")

train_images_scaled = scaler.fit_transform(train_images)
test_images_scaled = scaler.transform(test_images)

classifier = KNeighborsClassifier(n_neighbors=best_k, weights=best_weight, metric=best_metric)
classifier.fit(train_images_scaled, train_labels)
predictions_test = classifier.predict(test_images_scaled)

# save la submisie
test_df["label"] = predictions_test
nume_fisier = "submission_knn_" + best_metric + "_" + best_weight + "_k" + str(best_k) + ".csv"
test_df[["id", "label"]].to_csv(nume_fisier, index=False)
print("best:", nume_fisier)