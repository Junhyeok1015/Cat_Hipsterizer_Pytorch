import time
import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import MobileNetV2
import numpy as np
from dataload import *
from model import *
from utils import *

# switch to PyTorch
# TensorBoard -> replace with neptune
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = None # path to model checkpoint, None if none
# img_size = 224
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
workers = 4

mode = 'bbs' # [bbs, lmks]
if mode is 'bbs':
  output_size = 4
elif mode is 'lmks':
  output_size = 18

def main():
  """
  Training
  """
  # global start_epoch, epoch, checkpoint
  global checkpoint, output_size

  # Initialize model or load checkpoint
  if checkpoint is None:
    # starting from 0
    start_epoch = 0
    # define new model # 모델에 만들기!
    model = cat_hipsterizer_model(output_size)
    # define optimizer ###################### 옵티마이저 짜기!
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001 )
    # print("4325924593495")
  else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
  
  # Move to default device
  model = model.to(device)
  criterion = torch.nn.MSELoss()

  #dataset load from dataload.py
  train_dataset = get_train_dataset()
  test_dataset = get_test_dataset()

  print(train_dataset)

  training_epoch = 50

  print(start_epoch, training_epoch)

  # Training Model
  for epoch in range(start_epoch, training_epoch):
        print("Training!")
        
        # One epoch's training
        train(train_loader = train_dataset,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch = epoch)

        # Save checkpoint
        save_checkpoint_bbs(epoch, model, optimizer)

def train(train_loader, model, criterion, optimizer, epoch):
      """
      One epoch's training
      """

      model.train() # training mode enables dropout
      total_batch = len(train_loader)

      start = time.time()
      avg_cost = 0

      # Batches
      for i, (X, Y) in enumerate(train_loader):
            # Move to default device
            X = X.to(device)
            Y = Y.to(device)

            # Forward prop
            hypothesis = model(X)

            # Loss
            loss = criterion(hypothesis, Y)

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            avg_cost += loss/total_batch

            # Print Status
            print("Epoch: [{0}][{1}/{2}]\t"
            "Loss {3:.4f}\t".format(epoch, i, len(train_loader), avg_cost))


  # # training
  # model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

  # model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True,
  #   validation_data=(x_test, y_test), verbose=1,
  #   callbacks=[
  #     TensorBoard(log_dir='logs/%s' % (start_time)),
  #     ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
  #     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  #   ]
  # 
if __name__ == '__main__':
  main()