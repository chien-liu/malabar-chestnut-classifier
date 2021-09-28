'''
loss-epoch and accuracy-epoch is save in 'rec_loss.npy'
This file plot the diagrams.
'''

import numpy as np
import matplotlib.pyplot as plt


rec_loss = np.load('rec_loss.npy')

loss = [rl[0] for rl in rec_loss]
acc = [rl[1] for rl in rec_loss]

plt.plot(loss)
# plt.plot(acc)
plt.ylabel('Loss', size=20)
plt.xlabel('epoch', size=20)
plt.title('Transfer Learning -- Vgg19', size=20)
plt.tight_layout()
# plt.savefig('loss-epoch', dpi=300)
plt.show()



