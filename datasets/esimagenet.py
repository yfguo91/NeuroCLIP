import os
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

import h5py
import numpy as np

from spikingjelly.datasets.es_imagenet import ESImageNet

from IPython import embed

@DATASET_REGISTRY.register()
class ESIMAGENET_DATA(DatasetBase):

    def __init__(self, cfg):
        
        self.dataset_dir = cfg.DATASET.ROOT

        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)
        #embed()
        #train_data = ESImageNet(self.dataset_dir, train=True, data_type='frame', frames_number=cfg.MODEL.PROJECT.NUM_TIMES, split_by='number')
        #test_data = ESImageNet(self.dataset_dir, train=False, data_type='frame', frames_number=cfg.MODEL.PROJECT.NUM_TIMES, split_by='number')
        #embed()
        #train = self.read_data(classnames, train_data)
        #test = self.read_data(classnames, test_data)
        train = self.load_train_data(classnames, os.path.join(self.dataset_dir, 'train'))
        test = self.load_test_data(classnames, os.path.join(self.dataset_dir, 'test'))
        
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
        
    def load_train_data(self, classnames, data_path):
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
                    #embed()
                    item = Datum(
                        impath=data['frames'],
                        classname=classnames[int(data['label'])],
                        label=int(data['label'])
                    )
                    items.append(item)
        return items
    def load_test_data(self, classnames, data_path):
        # all_data = []
        # all_label = []
        items = []
        list_files = os.listdir(data_path)
        for file in list_files:
            data = np.load(os.path.join(data_path, file))
            # all_data.append(data['frame'])
            # all_label.append(self.keys[label])
            item = Datum(
                impath=data['frames'],
                classname=classnames[int(data['label'])],
                label=int(data['label'])
            )
            items.append(item)
        return items