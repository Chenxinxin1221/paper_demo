import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from torch.utils.data import DataLoader

def get_loader(data_path, batch_size, N_train, mode='train'):
    dataset = CardioLoader(data_path, N_train, mode)
    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

def relative_euclidean_distance(a, b):
    return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

class CardioLoader(object):
    def __init__(self, data_path, N_train, mode="train"):
        self.mode = mode
        data = np.load(data_path)
        np.set_printoptions(threshold=np.inf)
        index = (data == 1)
        data[index] = 0
        index = (data == -1)
        data[index] = 1
        labels = data[:, -1]
        features = data[:, :-1]
        normal_data = features[labels == 1]
        normal_labels = labels[labels == 1]
        attack_data = features[labels == 0]
        attack_labels = labels[labels == 0]
        N_attack = attack_data.shape[0]
        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        self.N_train = N_train
        self.train = attack_data[randIdx[:self.N_train]]
        self.train_labels = attack_labels[randIdx[:self.N_train]]
        self.test = attack_data[randIdx[self.N_train:]]
        self.test_labels = attack_labels[randIdx[self.N_train:]]
        self.test = np.concatenate((self.test, normal_data), axis=0)
        self.test_labels = np.concatenate((self.test_labels, normal_labels), axis=0)

    def __len__(self):

        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
            return np.float32(self.test[index]), np.float32(self.test_labels[index])

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enc_1 = nn.Linear(22, 20)
        self.enc = nn.Linear(20, 15)

        self.act = nn.Tanh()
        self.act_s = nn.Sigmoid()
        self.mu = nn.Linear(15, 15)
        self.log_var = nn.Linear(15, 15)

        self.z = nn.Linear(15, 15)
        self.z_1 = nn.Linear(15, 20)
        self.dec = nn.Linear(20, 22)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_1 = self.enc_1(x)
        enc = self.act(enc_1)
        enc = self.enc(enc)
        enc = self.act(enc)

        mu = self.mu(enc)
        log_var = self.log_var(enc)
        o = self.reparameterize(mu, log_var)
        z = self.z(o)
        z_1 = self.act(z)
        z_1 = self.z_1(z_1)
        dec = self.act(z_1)
        dec = self.dec(dec)
        dec = self.act_s(dec)
        return enc_1, enc, mu, log_var, o, z, z_1, dec


data_path = 'Cardiotocography.npy'

batch_size = 64
learn_rate = 0.0001
All_train = 1648

Ratio = 0.1
iter_per_epoch = 1000
Average_cycle = 4
result = []
diff_quantity_result = []
for i in range(6,7):
    N_train = int(All_train * Ratio * (i+2))      # 把8改为i+2
    result = []
    # print(Ratio * (i+2))
    for i in range(1):
        x_all = []
        z_all = []
        z1_all = []
        vae = torch.load('./model.pth')
        batch_size = 1000
        data_loader_train = get_loader(data_path, N_train, N_train, mode='train')
        train_enc = []
        train_labels = []
        data_loader_test = get_loader(data_path, batch_size, N_train, mode='test')
        test_enc = []
        test_labels = []

        for i, (input_data, labels) in enumerate(data_loader_train):
            enc_1, enc, mu, log_var, o, z, z_1, dec = vae(input_data)
            rec_euclidean = relative_euclidean_distance(input_data, dec)
            enc = torch.cat([enc, rec_euclidean.unsqueeze(-1)], dim=1)
            enc = enc.detach().numpy()

            train_enc.append(enc)
        for i, (input_data, labels) in enumerate(data_loader_test):
            enc_1, enc, mu, log_var, o, z, z_1, dec = vae(input_data)
            rec_euclidean = relative_euclidean_distance(input_data, dec)
            enc = torch.cat([enc, rec_euclidean.unsqueeze(-1)], dim=1)
            enc = enc.detach().numpy()

            test_enc.append(enc)
            test_labels.append(labels.numpy())
        x = train_enc[0]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.000001).fit(x)
        score = kde.score_samples(x)
        k = len(test_enc)
        test_score = []
        for i in range(k):
            score = kde.score_samples(test_enc[i])
            test_score.append(score)
        test_labels = np.concatenate(test_labels, axis=0)
        test_score = np.concatenate(test_score, axis=0)
        s = len(test_labels)
        c = np.sum(test_labels == 1)
        g = c / s
        np.set_printoptions(threshold=np.inf)
        thresh = np.percentile(test_score, int(g * 100))
        pred = (test_score < thresh).astype(int)
        gt = test_labels.astype(int)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')
        temp_result = [accuracy, precision, recall, f_score]
        result.append(temp_result)

    end_result = np.mean(result, axis=0)
    print(end_result)