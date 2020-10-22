# Initialize configuration for project including that
# creating directories
# setting cuda device order
import os
import matplotlib

from constant import args, TMP_DIR
from os.path import isdir

matplotlib.use('Agg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)
