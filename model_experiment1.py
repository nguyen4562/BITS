import torch
import cv2
import os
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import face_recognition

"""
    - list
        0 -> filter
        1 -> kernel
        2 -> stride
        3 -> padding
    - Tuple
        0 -> kernel
        1 -> stride
        2 -> padding
"""
config = [
    [64, 3, 1, 1],
    [64, 3, 1, 1],
    (2, 2, 0),
    [128, 3, 1, 1],
    [128, 3, 1, 1],
    (2, 2, 0),
    [256, 3, 1, 1],
    [256, 3, 1, 1],
    [256, 3, 1, 1],
    (2, 2, 0),
    [512, 3, 1, 1],
    [512, 3, 1, 1],
    [512, 1, 1, 0],
    (2, 2, 0),
    [512, 1, 1, 0],
    [512, 1, 1, 0],
    (2, 2, 0)
]


class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding)
        self.btch = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.btch(self.conv(x)))


class Net(nn.Module):
    def __init__(self, classes=1):
        super(Net, self).__init__()
        self.classes = classes
        self.in_channels = 3
        self.layers = self._make_layers(config)
        self.fc = self._make_fc()

        # self.cov1 = nn.Conv2d(3, )
        # self.linear1 = nn.Linear(25088, 128)

    def _make_layers(self, config):
        layers = []
        in_channels = self.in_channels

        for layer in config:
            if isinstance(layer, list):
                layers += [CNN(in_channels, layer[0], layer[1], layer[2], layer[3])]
                in_channels = layer[0]
            else:
                layers += [nn.MaxPool2d(layer[0], layer[1], layer[2])]
        return nn.Sequential(*layers)

    def _make_fc(self):
        return nn.Sequential(
            nn.Flatten(),
            # 25088
            # 86528
            nn.Linear(25088, 128),
            nn.LayerNorm(128)
        )

    def forward(self, x):
        return self.fc(self.layers(x))


class Data(Dataset):
    def __init__(self, csv_file, img_path, lbl_path=None):
        self.data = pd.read_csv(csv_file)
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.size = len(self.data)

        # self.images, self.labels = self._create()

    def _create(self):
        names = os.listdir(self.img_path)
        n_labels = self.size
        y = torch.arange(n_labels)
        x = torch.zeros((1, 3, 224, 224))
        label = {}
        batch = 0
        i = 0
        for name in names:
            label[name] = []
            batch = 0
            for image in os.listdir(os.path.join(self.img_path, name)):
                image_file = os.path.join(self.img_path, name, image)
                image = TF.to_tensor(cv2.imread(image_file)).unsqueeze(0)
                label[name].append(image)
                if i == 0:
                    x += image
                    i += 1
                else:
                    x = torch.concat([x, image])
                batch += 1
        x = x.reshape(n_labels, batch, 3, 224, 224)
        return x, y

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.data.iloc[index, 0])
        image = TF.to_tensor(Image.open(img_path))
        label = torch.tensor(self.data.iloc[index, 1])

        return image, label


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2.0, mining="batch_all"):
        super().__init__()
        self.margin = margin
        self.p = p
        self.mining = mining

        if mining == "batch_all":
            self.loss_fn = self.batch_all_triplet_loss

    def forward(self, embeddings, labels):
        return self.loss_fn(labels, embeddings, self.margin, self.p)

    def _get_triplet_mask(self, labels):
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels & distinct_indices

    def batch_all_triplet_loss(self, labels, embeddings, margin, p):
        pairwise_dist = torch.cdist(embeddings, embeddings, p=p)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        mask = self._get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets


def setup(csv_file, img_dirr, size):
    data = Data(csv_file, img_dirr)
    train_data = DataLoader(dataset=data, batch_size=size)
    return train_data


def train(x, y, epochs=-1, load=False, save=False):
    device = "cpu"
    model = Net(2)
    model.to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load Model from pth.tar file
    if load:
        check = torch.load("test.pth.tar")
        model.load_state_dict(check['model_state_dict'])
        optimizer.load_state_dict(check['optimizer_state_dict'])
        model.train()

    if epochs >= 1:
        for epoch in range(epochs):
            labels = y
            batch = x.shape[1]
            for label in range(len(labels)):
                for i in range(batch):
                    losses = []
                    for n in range(len(labels)):
                        if label != n:
                            for j in range(batch):
                                if i != j:
                                    anc = model(x[label, i].unsqueeze(0))
                                    pos = model(x[label, j].unsqueeze(0))
                                    neg = model(x[n, j].unsqueeze(0))

                                    loss = criterion(anc, pos, neg)
                                    losses.append(loss)

                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                                j += 1
                    print(sum(losses) / len(losses))
    if save:
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, "test.pth.tar")

    return model


def scale(image, name, size):
    cv2.imwrite(name, cv2.resize(image, (size, size)))


def overwrite(from_folder, to_folder, names, size):
    for name in names:
        for i in range(1, names[name] + 1):
            image = cv2.imread(from_folder + "/" + name + "-" + str(i) + ".png")
            image = cv2.resize(image, (size, size))
            cv2.imwrite(to_folder + "/" + name + "-" + str(i) + ".png", image)


def load_images(img_path, csv_path):
    x = torch.zeros((1, 3, 224, 224))
    n_labels = len(pd.read_csv(csv_path))
    y = torch.arange(n_labels)
    batch = 0
    i = 0
    for person in os.listdir(img_path):
        batch = 0
        for image in os.listdir(os.path.join(img_path, person)):
            image_file = os.path.join(img_path, person, image)
            img_rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
            image = TF.to_tensor(img_rgb).unsqueeze(0)
            if i == 0:
                x += image
                i += 1
            else:
                x = torch.concat([x, image])
            batch += 1
    x = x.reshape(n_labels, batch, 3, 224, 224)
    return x, y


def train1(data, epochs=-1, load=False, save=False):
    device = "cpu"
    model = Net(2)
    model.to(device)
    # criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = TripletMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load Model from pth.tar file
    if load:
        check = torch.load("test.pth.tar")
        model.load_state_dict(check['model_state_dict'])
        optimizer.load_state_dict(check['optimizer_state_dict'])
        model.train()

    if epochs >= 1:
        prv_loss = []
        for e in range(epochs):
            losses = []
            for i, (x, y) in enumerate(data):
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss, _ = criterion(output, y)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            print(f"{e + 1} - Loss: {mean_loss:.4g}")
            if e == 0:
                prv_loss.append(mean_loss)
            if save:
                if mean_loss <= min(prv_loss):
                    checkpoint = {'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict()}
                    torch.save(checkpoint, "test.pth.tar")
                    print(f"==> Save checkpoint! Loss {min(prv_loss):.4g}-{mean_loss:.4g}")
            prv_loss.append(mean_loss)

    return model


def predict(x, y, model, sample, image_file, viewAll=True):
    img_rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    image = TF.to_tensor(img_rgb).unsqueeze(0)
    result = model(image)

    dist = torch.norm(result - sample, dim=1, p=None)
    knn = dist.topk(1, largest=False)
    idx = torch.div(knn.indices, x.shape[0] // y.shape[0], rounding_mode='trunc')
    if viewAll:
        dist = dist.reshape(y.shape[0], x.shape[0] // y.shape[0])
        return idx, knn.values[0], dist[idx]
    else:
        return idx, knn.values[0]


def predictAll(x, y, model, sample, img_path, newData=None, start=0, stop=0):
    for person in os.listdir(img_path):
        i = 1
        for image in os.listdir(os.path.join(img_path, person)):
            image_file = os.path.join(img_path, person, image)
            idx, values = predict(x, y, model, sample, image_file, viewAll=False)
            print(i, person, name[idx], idx)
            i += 1
        print()
    for i in range(start, stop):
        idx, values = predict(x, y, model, sample, newData + str(i) + ".png", viewAll=False)
        print(i, name[idx], values)



def predict1(model, sample, number, label, image_file, viewAll=True):
    img_rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    image = TF.to_tensor(img_rgb).unsqueeze(0)
    result = model(image)

    dist = torch.norm(result - sample, dim=1, p=None)
    knn = dist.topk(1, largest=False)
    idx = torch.div(knn.indices, 45 // label, rounding_mode='trunc')
    if viewAll:
        dist = dist.reshape(label, 45 // label)
        return idx, knn.values[0], dist[idx]
    else:
        dist = dist.reshape(label, 45 // label)
        return idx, knn.values[0]
def predictAll1(model, sample, name, number, label, img_path):
    for person in os.listdir(img_path):
        i = 1
        for image in os.listdir(os.path.join(img_path, person)):
            image_file = os.path.join(img_path, person, image)
            idx, values = predict1(model, sample, number, label,image_file, viewAll=False)
            print(i, person, name[idx], idx)
            i += 1
        print()

def write_face(from_file, to_folder, infront, start, size, rmv=False):
    frame = cv2.imread(from_file)
    locations = face_recognition.face_locations(frame)
    i = start
    for location in locations:
        x = location[3]
        y = location[0]
        w = location[1] - x
        h = location[2] - y
        result = frame[y:y + h, x:x + w]
        cv2.imwrite(to_folder + infront + str(i) + ".png", cv2.resize(result, (size, size)))
        i += 1
        print("Detect Face!")
    if rmv:
        os.remove(from_file)


def randomBatch(num_random, batch, num_label):
    choices = torch.zeros((num_label, num_random), dtype=torch.int8)

    for i in range(0, num_label):
        images = torch.arange(batch).tolist()
        for j in range(num_random):
            choice = torch.randint(0, len(images), (1, 1))[0][0]
            choices[i][j] = images[choice]
            images[choice] = images[-1]
            images.pop()

    return choices


def writePrediction(current_csv, new_csv, label_csv, num_random):
    """
    Write all of random images per label into a csv file. Use that file for prediction (use Dataloader for efficiency).
    :param current_csv: the current csv file with all images of all labels
    :param new_csv: write image name to the new csv file.
    :param label_csv: the csv file contains labels
    :param num_random: number of images to write into the new_csv file
    :return: None
    """
    data = pd.read_csv(current_csv)
    lbls = pd.read_csv(label_csv)

    num_lbl = len(lbls)
    num_data = len(data)

    batch = num_data // num_lbl
    choices = randomBatch(num_random, batch, num_lbl)

    file = open(new_csv, 'w')
    file.write("images,label\n")
    for i in range(choices.shape[0]):
        for j in range(choices.shape[1]):
            file.write(f"{lbls.iloc[i, 1]}-{choices[i][j]+1}.png,{i}\n")


def takeSample(predict_csv, images_folder, batch):
    data = setup(predict_csv, images_folder, batch)
    device = "cpu"
    model = Net(2)
    model.to(device)
    sample = torch.zeros((1, 128))
    for i, (x, y) in enumerate(data):
        x = x.to(device)
        output = model(x)
        sample = torch.cat([sample, output], dim=0)
    sample = sample[1:]
    return sample


data = setup("data/train.csv", "data/temp", 24)
model = train1(data, epochs=0, load=True, save=False)
model.eval()
# print(sum(p.numel() for p in model.parameters()))

# x, y = load_images("data/images", "data/labels.csv")
# choices = torch.randint(0, x.shape[1], (x.shape[0], 3))
# temp = torch.zeros((1, 3, 224, 224))
# for i in range(x.shape[0]):
#     for j in range(3):
#         temp = torch.cat([ temp, x[i, choices[i, j]].unsqueeze(0) ])
# x = temp[1:]
# sample = model(x)
# name = ['Clara Nguyen', 'Dang', 'Nguyen', 'Unknown']
#
# # idx, value, dist = predict(x, y, model, sample, "data/images/Clara Nguyen/clara-nguyen-10.png", True)
# # print(idx)
# predictAll(x, y, model, sample, "data/images", newData="predict/All/all-", start=1, stop=7)




img_path = 'data/images'

data = pd.read_csv('data/train.csv')
lbls = pd.read_csv('data/labels.csv')

num_lbl = len(lbls)
num_data = len(data)
num_random = 15

batch = num_data // num_lbl
choices = randomBatch(num_random, batch, num_lbl)
x = torch.zeros((1, 3, 224, 224))
z = 0
k = 0
for i in range(num_lbl):
    for j in range(num_random):
        image_file = os.path.join(img_path, lbls.iloc[i, 0], data.iloc[z + int(choices[i][j]), 0])
        img_rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        image = TF.to_tensor(img_rgb).unsqueeze(0)
        if k == 0:
            x += image
            k += 1
        else:
            x = torch.concat([x, image])
    z += batch

x = x.to("cpu")
sample = model(x)
# name = ['Dang', 'Nguyen', 'Clara Nguyen', 'Unknown']
# y = torch.arange(3)
# predictAll("data/images", sample, newData="predict/All/all-", start=1, stop=7)




# data = setup("data/train.csv", "data/temp", 24)
# model = train1(data, epochs=0, load=True, save=False)
# model.eval()
#
# # writePrediction('data/train.csv', 'data/predict.csv', 'data/labels.csv', 15)
# sample = takeSample("data/predict.csv", 'data/temp', 24)
# frame = pd.read_csv('data/labels.csv')
#
# columns = [column for column in frame]
# names = [name for name in frame[columns[0]]]
# names = ['Clara Nguyen', "Dang", "Nguyen"]
#
# # idx, value = predict(model, sample, 3, "data/temp/clara-nguyen-7.png", viewAll=False)
# predictAll1(model, sample, names, 15, 3, "data/images")



# for i in range(1, 25):
#     names = ["Dang", "Nguyen"]
#     img_rgb = cv2.cvtColor(cv2.imread("data/temp/dang-" + str(i) + ".png"), cv2.COLOR_BGR2RGB)
#     image = TF.to_tensor(img_rgb).unsqueeze(0)
#     result = model(image)
#
#     dist = torch.norm(result - sample, dim=1, p=None)
#     knn = dist.topk(1, largest=False)
#     idx = torch.div(knn.indices, 15, rounding_mode='trunc')
#
#     print(knn.indices, idx)
#
# print()
#
# for i in range(1, 25):
#     names = ["Dang", "Nguyen"]
#     img_rgb = cv2.cvtColor(cv2.imread("data/temp/nguyen-" + str(i) + ".png"), cv2.COLOR_BGR2RGB)
#     image = TF.to_tensor(img_rgb).unsqueeze(0)
#     result = model(image)
#
#     dist = torch.norm(result - sample, dim=1, p=None)
#     knn = dist.topk(1, largest=False)
#     idx = torch.div(knn.indices, 15, rounding_mode='trunc')
#
#     print(knn.indices, idx)





# j = [5, 7]
# for i in range(2):
#     write_face("data/images/Dang/" + str(j[i]) + ".png",
#                "data/images/Dang/",
#                "dang-",
#                j[i], 224, True)






# import face_recognition

# data = setup("data/train.csv", "data/temp", 24)
# model = train1(data, epochs=0, load=True, save=False)
# model.eval()


# x, y = load_images("data/images", "data/train-2.csv")
# choices = torch.randint(0, x.shape[1], (x.shape[0], 3))
# temp = torch.zeros((1, 3, 224, 224))
# for i in range(x.shape[0]):
#     for j in range(3):
#         temp = torch.cat([ temp, x[i, choices[i, j]].unsqueeze(0) ])
# x = temp[1:]
# sample = model(x)



# cap = cv2.VideoCapture(0)
# while True:
#
#     _, frame = cap.read()
#     locations = face_recognition.face_locations(frame)
#
#     if len(locations) != 0:
#         idx = -1
#         for location in locations:
#             x1 = location[3]
#             y1 = location[0]
#             w = location[1] - x1
#             h = location[2] - y1
#             predc = TF.to_tensor(cv2.resize(frame[y1:y1 + h, x1:x1 + w], (224, 224))).unsqueeze(0)
#
#             output = model(predc)
#             dist = torch.norm(output - sample, dim=1, p=None)
#             knn = dist.topk(1, largest=False)
#             idx = torch.div(knn.indices, num_random, rounding_mode='trunc')
#
#             # if knn.values[0] > 0.5:
#             #     idx = -1
#             #     print(knn.values[0])
#             print(knn.values[0])
#
#             top, right, bottom, left = location
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
#             cv2.rectangle(frame, (left, bottom - 17), (right, bottom), (0, 0, 255), cv2.FILLED)
#             cv2.putText(frame, name[idx], (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#
#     cv2.imshow("TESTING", frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break