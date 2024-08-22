from .utils import Datum
import os
template = ['a photo of {}.']
from parse_args import parse_args

parser = parse_args()
args = parser.parse_args()



def read_data(filepath):
    
    with open(filepath, 'r') as f:
        out = []
        for line in f.readlines():
            img_path, label = line.strip().split(' ')
            impath = os.path.join(args.root_path, img_path)
            item = Datum(
            impath=impath,
            label=int(label)
            )
            out.append(item)
        return out
        
def read_data_t(filepath):
    
    with open(filepath, 'r') as f:
        out = []
        for line in f.readlines():
            img_path, label = line.strip().split(' ')
            item = Datum(
            impath=img_path,
            label=int(label)
            )
            out.append(item)
        return out
        

class TrainSet_double():

    def __init__(self):

        self.template = template
        trainx = read_data('caches/train.txt')
        valx = read_data('caches/val.txt')
        
        if os.path.exists('caches/temp_labels.txt'):
            trainx_double = read_data_t('caches/temp_labels.txt')
            trainx_double_paths = set(item.impath for item in trainx_double)
            self.train_x = trainx + trainx_double
            # self.val =  [item for item in valx if item.impath not in trainx_double_paths]
            self.val =valx
        else :
            self.train_x = trainx
            self.val =valx
        self.test = read_data('caches/testb.txt')