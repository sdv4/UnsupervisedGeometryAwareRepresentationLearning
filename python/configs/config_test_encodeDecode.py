import os

from utils import io as utils_io

config_dict = utils_io.loadModule("./configs/config_train_encodeDecode_pose.py").config_dict
config_dict['num_workers'] = 0
config_dict['label_types_test'].remove('img_crop')
config_dict['label_types_train'].remove('img_crop')
config_dict['batch_size_train'] = 1
config_dict['batch_size_test'] = 1

network_path = '../examples'
config_dict['pretrained_network_path'] = network_path + '/network_best_val_t1.pth'
if not os.path.exists(config_dict['pretrained_network_path']):
    import urllib.request
    print("Downloading pre-trained weights, can take a while...")
    urllib.request.urlretrieve("http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/ECCV2018Rhodin/network_best_val_t1.pth",
                               config_dict['pretrained_network_path'])
    print("Downloading done.")
