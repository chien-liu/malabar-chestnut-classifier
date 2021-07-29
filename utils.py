import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf

class BatchGenerator:
	def __init__(
			self,
			batch_size=32,
			path_data1='./../photo/dataset/',
			path_data2='./../trainData/',
			):
		self.batch_size = batch_size
		self.path_data1 = path_data1
		self.path_data2 = path_data2
		self._dataset = pd.DataFrame(columns=['name', 'positive', 'negative', 'medium'])

		# search data from paths
		self.search_data1(path_data1)
		self.search_data2(path_data2)

		self.generate_batch(self.batch_size)

	def search_data1(self, path):
		files = ['positive', 'negative', 'medium']
		ground_truths = [[1,0,0], [0,1,0,], [0,0,1]]
		
		names = []
		labels = []
		for file, label in zip(files, ground_truths):
			name = os.listdir(path+file)
			full_name = [path+file+'/'+n for n in name]
			n = len(name)
			names.extend(full_name)

			labels.append(np.array(label*n).reshape(-1,3))

		labels = np.vstack(labels)

		self._dataset['name'] = names
		self._dataset.iloc[:, 1:] = labels
		
	def search_data2(self, path):
		files = [f for f in os.listdir(path) if os.path.isdir(path+f)]

		# get exist csv
		arr_ls = []
		for file in files:
			csvFile = path+file+'/'+'label.csv'
			if os.path.exists(csvFile):
				arr_ls.append(pd.read_csv(csvFile).values)
		arr = np.vstack(arr_ls)
		
		# get name
		name = arr[:,0]
		full_name = [path + n for n in name]
		# get label
		label = arr[:, 1]
		one_hot_label = np.zeros((len(full_name), 3))
		one_hot_label[(label=='u').ravel(), 0] = 1
		one_hot_label[(label=='d').ravel(), 1] = 1
		one_hot_label[(label=='m').ravel(), 2] = 1
		df = pd.DataFrame({'name': full_name,
							'positive': one_hot_label[:,0],
							'negative': one_hot_label[:,1],
							'medium': one_hot_label[:,2]},
							columns=['name', 'positive', 'negative', 'medium'])
		self._dataset = self._dataset.append(df, ignore_index=True)

	def get_data(self,):
		return self._dataset

	def shuffle(self,):
		self._dataset = self._dataset.sample(n=len(self._dataset))

	def generate_batch(self, batch_size):
		self.shuffle()
		data = self._dataset.values
		self.batch_xs, self.batch_ys = [], []

		for i in range(len(self._dataset) // batch_size):

			# generate batch_x
			batch_x = data[i * batch_size : (i+1) * batch_size, 0]

			# generate batch_y
			batch_y = data[i * batch_size : (i+1) * batch_size, 1:]

			self.batch_xs.append(batch_x)
			self.batch_ys.append(batch_y)

	def len(self,):
		return len(self._dataset) // self.batch_size

	def get(self, batch_id):

		return self.distort_input(self.batch_xs[batch_id]), self.batch_ys[batch_id]


	def distort_input(self, training_files):
		""" Construct distorted input for CIFAR training using the Reader ops.
			-----
			Args:
					training_files: 
							an array of paths of the training files.
					batch_size: 
							Number of images per batch.
			Returns:
					images: Images. 
							4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		"""
		for f in training_files:
			if not os.path.isfile(f):
				raise ValueError('Failed to find file: ' + f)

		def random_crop(img, crop_size=4/5):
			crop_size = np.random.uniform(crop_size, 1)
			height, width = img.shape[:2]
			h, w = int(height*crop_size), int(width*crop_size)
			y = np.random.choice(height-h)
			x = np.random.choice(width-w)
			crop_img = img[y:y+h, x:x+w]
			crop_img = cv2.resize(crop_img, (width, height))

			return crop_img
		
		def random_flip_left_right(img):
			number = np.random.choice([-1, 0, 1, 2])
			if number == 2:
				return img
			else:
				return cv2.flip(img, number)

		def random_brightness(img, delta=.25):
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
			random_bright = delta + np.random.uniform()
			if random_bright > 1:
				random_bright = 1 
			hsv[:,:,2] = hsv[:,:,2] * random_bright
			img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
			return img

		def random_contrast(img):
			return img

		distorted_images = []

		for f in training_files:
			img = cv2.imread(f, 1)
			distorted_image = cv2.resize(img, (224, 224))
			distorted_image = random_crop(distorted_image)
			distorted_image = random_flip_left_right(distorted_image)
			distorted_image = random_brightness(distorted_image)
			distorted_image = random_contrast(distorted_image)
			distorted_images.append(distorted_image)
		
		return np.array(distorted_images)

	


if __name__ == '__main__':
	BG = BatchGenerator()
	data = BG.get_data()
	n = BG.len()

	x, y = BG.get(10)
	
	print(y.shape)