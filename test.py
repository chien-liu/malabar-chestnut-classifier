import cv2
import pandas as pd
import numpy as np
from net import VGG19
import time
from statistics import mean
import random
from tqdm import tqdm


def test(model, testData_path='./../testData/'):
    FILES = 100
    PHOTOS = 60
    PHOTO_START = 30
    score = 0
    loose_score = 0

    img_time = []
    seed_time = []

    for files in tqdm(range(FILES)):
        df = pd.read_csv('{}{}/label.csv'.format(testData_path, files))

        labels = df['label'].values

        seed_start = time.time()
        photo_start = np.random.randint(PHOTO_START)
        for photo in range(photo_start, PHOTOS):
            img = cv2.imread('{}{}/{}.png'.format(testData_path, files, photo))

            label = labels[photo]

            start = time.time()
            predict = model.predict(img)
            stop = time.time()
            img_time.append(stop - start)

            # Grading
            if predict == 'd':
                if label == 'd':
                    score += 1
                    loose_score += 1
                elif label == 'm':
                    loose_score += 1
                break

        seed_stop = time.time()
        seed_time.append(seed_stop-seed_start)

    avg_img_time = mean(img_time)
    avg_seed_time = mean(seed_time)

    print('score:', score)
    print('loose_score:', loose_score)
    print('time per image:', avg_img_time)
    print('time per seed:', avg_seed_time)
    return score, loose_score, avg_img_time, avg_seed_time


class random_model:
    def __init__(self,):
        pass

    def predict(self, img):
        if img:
            return random.choice('udm')


def main():
    # randclassifier = random_model()
    vgg19 = VGG19(vgg19_npy_path='./weights/fine_tune_weight.npy')

    stricts, looses, img_times, seed_times = [], [], [], []
    for _ in range(10):
        strict, loose, img_time, seed_time = test(model=vgg19)
        stricts.append(strict)
        looses.append(loose)
        img_times.append(img_time)
        seed_times.append(seed_time)

    print('VGG19')
    print('score:', mean(stricts))
    print('loose_score:', mean(looses))
    print('fps:', 1/mean(img_times))
    print('time per seed:', mean(seed_times))


if __name__ == '__main__':
    main()
