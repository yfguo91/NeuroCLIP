import os
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

import h5py
import numpy as np

from spikingjelly.datasets.cifar10_dvs  import CIFAR10DVS
import spikingjelly

from IPython import embed

@DATASET_REGISTRY.register()
class CIFAR10DVS_DATA(DatasetBase):

    def __init__(self, cfg):
        
        self.dataset_dir = cfg.DATASET.ROOT

        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)
        
        data_set = CIFAR10DVS(self.dataset_dir, data_type='frame', frames_number=cfg.MODEL.PROJECT.NUM_TIMES, split_by='number')
        train_data, test_data = spikingjelly.datasets.split_to_train_test_set(0.9, data_set, 10, random_split = False)
        
        train = self.read_data(classnames, train_data)
        test = self.read_data(classnames, test_data)
        
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
    
    def read_data(self, classnames, datas):
        items = []
        
        for i, data in enumerate(datas):
            label = int(data[1])
            classname = classnames[label]

            item = Datum(
                impath=data[0],
                label=label,
                classname=classname
            )
            items.append(item)
        
        return items
