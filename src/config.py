import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.6
POS_WEIGHT = 4.5
LAMBDA_DICE = 1.0

PATIENCE = 10
EPOCHS = 40

BATCH_SIZE = 16

LR = 1e-4
WEIGHT_DECAY = 1e-5

RANDOM_SEED = 42

TEST_SIZE = 0.2

IMG_PATH = "../data/"
IMG_LIMIT = 100
