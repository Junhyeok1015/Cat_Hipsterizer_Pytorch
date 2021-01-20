import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    # Total data
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

img_size = 224
batch_size = 32
workers = 4

mode = 'bbs' # [bbs, lmks]
if mode is 'bbs':
  output_size = 4
elif mode is 'lmks':
  output_size = 18

# error occured // ValueError: Object arrays cannot be loaded when allow_pickle=False)
# save original np.load into np_load_old
np_load_old = np.load
# change parameter of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# DataLoader for BBS
def get_train_dataset_bbs():
    print("Train Dataload Start!")


    data_00 = np.load('dataset/CAT_00.npy')
    data_01 = np.load('dataset/CAT_01.npy')
    data_02 = np.load('dataset/CAT_02.npy')
    data_03 = np.load('dataset/CAT_03.npy')
    data_04 = np.load('dataset/CAT_04.npy')
    data_05 = np.load('dataset/CAT_05.npy')

    print("Train Dataload Finish!")
    print("Train Data Preprocessing Start!")

    x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
    y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)


    x_train = x_train.astype('float32') / 255.
    x_train = np.reshape(x_train, (-1, 3, img_size, img_size))

    y_train = np.reshape(y_train, (-1, output_size))

    print("Train Data Preprocessing Finish!")

    dataset = CustomDataset(x_train, y_train)

    train_dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = workers,  pin_memory=True)


    return train_dataloader

def get_test_dataset_bbs():
    print("Test Dataload Start!")

    data_06 = np.load('dataset/CAT_06.npy')

    print("Test Dataload Finish!")
    print("Test Data Preprocessing Start!")

    x_test = np.array(data_06.item().get('imgs'))
    y_test = np.array(data_06.item().get(mode))

    x_test = x_test.astype('float32') / 255.
    x_test = np.reshape(x_test, (-1, 3, img_size, img_size))

    y_test = np.reshape(y_test, (-1, output_size))

    print("Test Data Preprocessing Finish!")

    dataset = CustomDataset(x_test, y_test)

    test_dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = workers,  pin_memory=True)

    return test_dataloader



# DataLoader for LMKS
def get_train_dataset_lmks():
    print("Train Dataload Start!")

    data_00 = np.load('dataset/lmks_CAT_00.npy')
    data_01 = np.load('dataset/lmks_CAT_01.npy')
    data_02 = np.load('dataset/lmks_CAT_02.npy')
    data_03 = np.load('dataset/lmks_CAT_03.npy')
    data_04 = np.load('dataset/lmks_CAT_04.npy')
    data_05 = np.load('dataset/lmks_CAT_05.npy')

    print("Train Dataload Finish!")
    print("Train Data Preprocessing Start!")

    x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
    y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)


    x_train = x_train.astype('float32') / 255.
    x_train = np.reshape(x_train, (-1, 3, img_size, img_size))

    y_train = np.reshape(y_train, (-1, output_size))

    print("Train Data Preprocessing Finish!")

    dataset = CustomDataset(x_train, y_train)

    train_dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers = workers,  pin_memory=True)


    return train_dataloader

def get_test_dataset_lmks():
    print("Test Dataload Start!")

    data_06 = np.load('dataset/lmks_CAT_06.npy')

    print("Test Dataload Finish!")
    print("Test Data Preprocessing Start!")

    x_test = np.array(data_06.item().get('imgs'))
    y_test = np.array(data_06.item().get(mode))

    x_test = x_test.astype('float32') / 255.
    x_test = np.reshape(x_test, (-1, 3, img_size, img_size))

    y_test = np.reshape(y_test, (-1, output_size))

    print("Test Data Preprocessing Finish!")

    dataset = CustomDataset(x_test, y_test)

    test_dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = workers,  pin_memory=True)

    return test_dataloader