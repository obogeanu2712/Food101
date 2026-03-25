from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.Image import Resampling

import multiprocessing

SIZE = 224
CLASS_COUNTS = 101
TRAIN_COUNTS = 750
TEST_COUNTS = 250

def get_stats(path : str, size : int = SIZE) :
    img_path = Path('./food-101/images') / Path(path + str('.jpg'))
    img = Image.open(img_path).convert('RGB').resize(size = (size, size), resample=Resampling.BICUBIC)
    img = np.asarray(img, dtype=np.uint32) / 255
    return np.sum(img, axis=(0, 1)), np.sum(img ** 2, axis=(0, 1))


if __name__ == '__main__' :

    with open('./food-101/meta/train.txt') as train_paths_file :
        train_image_paths = [line.strip() for line in train_paths_file]

    with open('./food-101/meta/test.txt') as test_paths_file :
        test_image_paths = [line.strip() for line in test_paths_file]


    print('Compute train stats : ')

    with multiprocessing.Pool() as pool :
        results = np.array((list(tqdm(pool.imap_unordered(get_stats, train_image_paths), total=TRAIN_COUNTS * CLASS_COUNTS))))

    sum = np.sum(results[:, 0, :], axis=0)

    sum_sqr = np.sum(results[:, 1, :], axis=0)

    train_mean = sum / CLASS_COUNTS / TRAIN_COUNTS / SIZE / SIZE

    sum_sqr_mean = sum_sqr / CLASS_COUNTS / TRAIN_COUNTS / SIZE / SIZE

    train_std = np.sqrt(sum_sqr_mean - train_mean ** 2)

    print(f'Mean / Std : {train_mean}, {train_std}')

    print('Compute test stats : ')

    with multiprocessing.Pool() as pool :
        results = np.array((list(tqdm(pool.imap_unordered(get_stats, test_image_paths), total=TEST_COUNTS * CLASS_COUNTS))))

    # print(results.shape) ## (TEST_COUNTS * CLASS_COUNTS, 2, 3) 2 - sum, sqr, 3 - rgb

    sum = np.sum(results[:, 0, :], axis=0)

    sum_sqr = np.sum(results[:, 1, :], axis=0)

    test_mean = sum / CLASS_COUNTS / TEST_COUNTS / SIZE / SIZE

    sum_sqr_mean = sum_sqr / CLASS_COUNTS / TEST_COUNTS / SIZE / SIZE

    test_std = np.sqrt(sum_sqr_mean - test_mean ** 2)

    print(f'Mean / Std : {test_mean}, {test_std}')

    with open('stats.txt', 'a') as file :
        file.write(f'Train (mean - std) : {train_mean} - {train_std}\n')
        file.write(f'Test (mean - std) : {test_mean} - {test_std}\n')