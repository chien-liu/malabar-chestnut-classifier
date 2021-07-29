import cv2
import pandas as pd
import numpy as np
from net import Vgg19

vgg19 = Vgg19(vgg19_npy_path='vgg19_fine_tuning_enough.npy')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('vgg19_fine_tune_slow.avi',fourcc, 5.0, (640,480))

double_break = False
for files in range(26,30):
	df = pd.read_csv('./../testData/' + str(files) + '/' + 'label.csv')
	labels = df['label']
	for photo in range(60):
		img = cv2.imread('./../testData/' + str(files) + '/' + str(photo) + '.png')
		label = vgg19.predict(img)
		if label == 'd':
			cv2.rectangle(img,(0,0),(640,480),(0,255,0),20)
		elif label == 'u':
			cv2.rectangle(img,(0,0),(640,480),(0,0,255),20)
		elif label == 'm':
			cv2.rectangle(img,(0,0),(640,480),(0,100,100),20)

		out.write(img)
		img = cv2.resize(img, (640+320, 480+240))

		cv2.imshow(str(files), img)
		if cv2.waitKey(100) & 0xFF == ord('q'):
			double_break = True
			break
	cv2.destroyAllWindows()
	if double_break:
		break

out.release()