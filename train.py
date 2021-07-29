from utils import BatchGenerator
from net import Vgg19
import time
import numpy as np

#----------BatchGenerator----------#
BATCH_SIZE = 16
epoch = 5

BG = BatchGenerator(batch_size=BATCH_SIZE)
data = BG.get_data()
NUM_BATCH = BG.len()
print(NUM_BATCH, 'batches.')
#----------Load pretrained network----------#
vgg19 = Vgg19(vgg19_npy_path='./vgg19_fine_tuning.npy')

rec_loss = np.load('rec_loss.npy').tolist()
print('record loss loaded.')


for e in range(epoch):
	print('epoch', e)
	BG.generate_batch(batch_size=BATCH_SIZE)
	loss_train = 0
	accuracy_train = 0

	for i in range(NUM_BATCH):
		
		batch_x, batch_y = BG.get(i)
		loss_batch, accuracy_batch = vgg19.train(batch_x, batch_y, lr=1e-5, partial_train=False)

		loss_train += loss_batch
		accuracy_train += accuracy_batch

	loss_train /= NUM_BATCH
	accuracy_train /= NUM_BATCH

	rec_loss.append([loss_train, accuracy_train])
	
	# print('Loss:', [rl[0] for rl in rec_loss])
	# print('Accuracy', [rl[1] for rl in rec_loss])

	if e%1 == 0:
		np.save('rec_loss.npy', rec_loss)
		vgg19.save_npy('vgg19_fine_tuning.npy')
		print('Module saved.')