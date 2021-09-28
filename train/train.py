'''This code execute training process and save the new weight.
BatchGenerator search data in ./../data/train-data.
One can manipulate training process by changing arguments.

fine_tune=True => all weights will be back-propagated.
fine_tune=False => only the weights in the last fully-connected layer will be back-propagated.
When new data is added. I recommand to train with setting fine_tune=False.After the net converges
(loss no longer decrease), set fine_tune=True and train for 5 epoch. Fine tuning with too many epoch
could leads to overfitting. 

Note
If new_weight_path is the same as existing weight, new weight will cover the old one.
'''

# The code could wash out trained weight.
# Don't execute it unless one knows how to train VGG19 Net.
assert False, "Don't use it."

from utils import BatchGenerator
from net.vgg19 import VGG19

import time
import numpy as np
import os

#----------Arguements----------#
BATCH_SIZE = 16
epoch = 5   # training epoch
learning_rate = 1e-5
fine_tune = True
original_weight_path = './../weights/vgg19.npy'
new_weight_path = './../weights/fine_tune_weight_ver2.npy'

#----------BatchGenerator----------#
BG = BatchGenerator(batch_size=BATCH_SIZE)
data = BG.get_data()
NUM_BATCH = BG.len()
print(NUM_BATCH, 'batches.')

#----------Load pretrained network----------#
vgg19 = VGG19(vgg19_npy_path=original_weight_path)

if os.path.exists('rec_loss.npy'):
    rec_loss = np.load('rec_loss.npy').tolist()
    print('record loss loaded.')
else:
    rec_loss = []


for e in range(epoch):
    start = time.time()

    print('epoch', e)
    BG.generate_batch(batch_size=BATCH_SIZE)
    loss_train = 0
    accuracy_train = 0

    for i in range(NUM_BATCH):
        
        batch_x, batch_y = BG.get(i)
        loss_batch, accuracy_batch = vgg19.train(batch_x, batch_y, lr=learning_rate, fine_tune=fine_tune)

        loss_train += loss_batch
        accuracy_train += accuracy_batch

    loss_train /= NUM_BATCH
    accuracy_train /= NUM_BATCH

    rec_loss.append([loss_train, accuracy_train])

    end = time.time()
    print('{}th epoch.\nTime spent {} sec.'.format(e+1, end-start))

    if e%1 == 0:
        # np.save('rec_loss.npy', rec_loss)
        vgg19.save_npy('./../net/vgg19-save.npy')
        print('Module saved.')