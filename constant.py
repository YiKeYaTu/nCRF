import argparse

from os.path import abspath, join, dirname


parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--lr', '--learning_rate', default=1e-8, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='nCRF')
parser.add_argument('--dataset', help='root folder of dataset',
                    default='../../../Dataset/HED-BSDS_PASCAL')
parser.add_argument('--enable_pretrain', help='Enable pretrained network',
                    type=bool, required=True, default=False)

args = parser.parse_args()

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, '__tmp__', args.tmp)
TEST_LIST_DIR = join(THIS_DIR, '../../../Dataset/HED-BSDS_PASCAL/test.lst')
