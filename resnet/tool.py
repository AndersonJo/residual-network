import os
import pickle
import tarfile
from urllib.request import urlopen

import numpy as np

# Constants
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def _download(dest_dir):
    """
    데이터가 없으면 CIFAR-10 데이터셋을 인터넷에서 다운로드한다.
    """
    tar_path = os.path.abspath(os.path.join(dest_dir, 'cifar-10-python.tar.gz'))

    # 저장할 디렉토리 생성
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # 파일이 존재하지 않으면 다운로드
    if not os.path.exists(tar_path):
        print('Start downloading CIFAR-10 dataset from internet')
        print(f'the path to save: {tar_path}')
        raw = urlopen(CIFAR_URL).read()
        with open(tar_path, 'wb') as f:
            f.write(raw)

    return tar_path


def _uncompress(data_path):
    """
    압축된 cifar-10-python.tar.gz 파일을 읽어서 data_path에 압축을 해제한다.
    """
    data_path = os.path.abspath(data_path)
    tar_path = os.path.join(data_path, 'cifar-10-python.tar.gz')
    uncompressed_path = os.path.join(data_path, 'cifar-10-batches-py')

    if not os.path.exists(uncompressed_path):
        print(f'Extracting {tar_path}')
        archive = tarfile.open(tar_path, 'r')
        archive.extractall(data_path)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def _to_onehot(data):
    N = data.shape[0]
    n_label = len(np.unique(data))

    onehot = np.zeros((N, n_label), dtype=np.int)
    onehot[np.arange(N), data] = 1
    return onehot


def _preprocessing1(data_path, force=False):
    """
    1차 과정으로 압축을 풀은 데이터 파일로 부터 읽어드린후, 각각의 파일을 합쳐서 새로운 파일로 저장을 한다.
    """
    abs_path = os.path.abspath(data_path)
    preprocessed_path = os.path.join(abs_path, 'cifar-10-preprocessed.pkl')
    uncompressed_dir = os.path.join(abs_path, 'cifar-10-batches-py')

    if os.path.exists(preprocessed_path) and not force:
        return preprocessed_path

    data_x = list()
    data_y = list()
    for i in range(1, 6):
        rawdata = unpickle(os.path.join(uncompressed_dir, f'data_batch_{i}'))
        data_x.append(rawdata[b'data'])
        data_y.append(rawdata[b'labels'])
    rawdata = unpickle(os.path.join(uncompressed_dir, 'test_batch'))

    train_x = np.array(data_x).reshape(-1, 3, 32, 32)
    train_x = train_x.transpose([0, 2, 3, 1])
    train_y = np.array(data_y).reshape(-1)

    test_x = np.array(rawdata[b'data']).reshape(-1, 3, 32, 32)
    test_x = test_x.transpose([0, 2, 3, 1])
    test_y = np.array(rawdata[b'labels']).reshape(-1)

    dataset = dict()
    dataset['train_x'] = train_x.astype('float32')
    dataset['train_y'] = train_y
    dataset['test_x'] = test_x.astype('float32')
    dataset['test_y'] = test_y

    f = open(preprocessed_path, 'wb')
    pickle.dump(dataset, f)

    return preprocessed_path


def load_data(data_path):
    _download(data_path)
    _uncompress(data_path)
    prep_path = _preprocessing1(data_path)

    dataset = pickle.load(open(prep_path, 'rb'))
    return dataset['train_x'], dataset['train_y'], dataset['test_x'], dataset['test_y']


def numpy_to_image(matrix: np.array, path: str):
    import scipy.misc
    scipy.misc.imsave(path, matrix)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data('_dataset')
    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print('test_x :', test_x.shape)
    print('test_y :', test_y.shape)
