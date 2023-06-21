import os
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

import h5py
import numpy as np

from spikingjelly.datasets.n_mnist import NMNIST

from IPython import embed

@DATASET_REGISTRY.register()
class NCARS(DatasetBase):

    def __init__(self, cfg, debug=False):
        
        if debug:
            self.dataset_dir = './ncars/' # input your path
        else:
            self.dataset_dir = cfg.DATASET.ROOT
        
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)
        self.keys = {'car':0, 'background':1}
        # train_data = NMNIST(self.dataset_dir, train=True, data_type='frame', frames_number=10, split_by='number')
        # test_data = NMNIST(self.dataset_dir, train=False, data_type='frame', frames_number=10, split_by='number')
        train = self.load_data(os.path.join(self.dataset_dir, 'train'))
        test = self.load_data(os.path.join(self.dataset_dir, 'test'))
        
        # train = self.read_data(classnames, train_data, train_label)
        # test = self.read_data(classnames, test_data, train_label)
        if debug:
            print('train_num:', len(train))
            print('test_num:', len(test))
        if debug:
            num_shots = 10
        else:
            num_shots = cfg.DATASET.NUM_SHOTS
            
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=test, test=test)


    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                classname = line.strip()
                classnames[i] = classname
        return classnames
    
    
    def load_data(self, data_path):
        # all_data = []
        # all_label = []
        items = []
        for root, dirs, files in os.walk(data_path):
            labels = dirs
            for label in labels:
                list_files = os.listdir(os.path.join(root, label))
                for file in list_files:
                    data = np.load(os.path.join(root, label, file))
                    # all_data.append(data['frame'])
                    # all_label.append(self.keys[label])
                    item = Datum(
                        impath=data['frame'],
                        classname=label,
                        label=self.keys[label]
                    )
                    items.append(item)
        return items
            # pass
        # return all_data, all_label
        
        
    # def read_data(self, classnames, datas, labels):
    #     items = []
        
    #     for i, data in enumerate(datas):
    #         label = int(labels[1])
    #         classname = classnames[label]

    #         item = Datum(
    #             impath=data[0],
    #             label=label,
    #             classname=classname
    #         )
    #         items.append(item)
        
    #     return items


if __name__ =='__main__':
    data = NCARS(cfg=None, debug=True)
    # pass
