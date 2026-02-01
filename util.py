import numpy as np
import os
import scipy.sparse as sp
import torch


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
    )
    # Data format
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])


    outdata = {}
    outdata['y_test'] = data['y_test']
    outdata["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    outdata["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
    outdata["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
    outdata["scaler"] = scaler

    return outdata

def get_dataloaders_from_index_data_MTS(
    data_dir,
    in_steps=12,
    out_steps=12,
    tod=True,
    dow=True,
    y_tod=False,
    y_dow=False,
    batch_size=64,
    log=None,
):
    raw_data = np.load(os.path.join(data_dir, f"data.npz"))["data"].astype(np.float32)
    index = np.load(os.path.join(data_dir, f"index_{in_steps}_{out_steps}.npz"))

    x_features = [0]
    if tod:
        x_features.append(1)
    if dow:
        x_features.append(2)

    y_features = [0]
    if y_tod:
        y_features.append(1)
    if y_dow:
        y_features.append(2)

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    # Iterative
    x_train = np.stack([raw_data[idx[0] : idx[1]] for idx in train_index])[..., x_features]
    y_train = np.stack([raw_data[idx[1] : idx[2]] for idx in train_index])[..., y_features]
    x_val = np.stack([raw_data[idx[0] : idx[1]] for idx in val_index])[..., x_features]
    y_val = np.stack([raw_data[idx[1] : idx[2]] for idx in val_index])[..., y_features]
    x_test = np.stack([raw_data[idx[0] : idx[1]] for idx in test_index])[..., x_features]
    y_test = np.stack([raw_data[idx[1] : idx[2]] for idx in test_index])[..., y_features]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    
   
    data = {}
    # data["x_train"], data["y_train"] = x_train, y_train
    # data["x_val"], data["y_val"] = x_val, y_val
    data["x_test"], data["y_test"] = x_test, y_test

    data["train_loader"] = DataLoader(x_train, y_train, batch_size)
    data["val_loader"] = DataLoader(x_val, y_val,  batch_size)
    data["test_loader"] = DataLoader(x_test, y_test, batch_size)
    data["scaler"] = scaler

    return data


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss


def metric(pred, real):
    mae = MAE_torch(pred, real, 0.0).item()
    mape = MAPE_torch(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    rmse = RMSE_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape
