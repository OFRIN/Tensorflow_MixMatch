IMAGE_SIZE = 32
IMAGE_CHANNEL = 3

CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
CLASSES = len(CLASS_NAMES)

CIFAR_10_MEAN = [0.32768, 0.32768, 0.32768] # [83.88608, 83.88608, 83.88608]
CIFAR_10_STD = [0.26811677, 0.26929596, 0.27755317] # [68.15831, 68.40918, 70.49192]

BATCH_SIZE = 64

MAX_ITERATION = 1024 * 2048
LOG_ITERATION = 1024 // 4
SAVE_ITERATION = 1024

NUM_THREADS = 1
