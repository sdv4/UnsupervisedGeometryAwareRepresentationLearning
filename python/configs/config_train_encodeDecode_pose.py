from utils import io as utils_io
import os

config_dict = utils_io.loadModule("./configs/config_train_encodeDecode.py").config_dict
config_dict['pretrained_network_path'] = '../examples/network_best_val_t1.pth'

if not os.path.exists(config_dict['pretrained_network_path']):
    import urllib.request
    print("Downloading pre-trained weights, can take a while...")
    urllib.request.urlretrieve("http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/ECCV2018Rhodin/network_best_val_t1.pth",
                               config_dict['pretrained_network_path'])
    print("Downloading done.")

config_dict['label_types_test'] += ['pose_mean', 'pose_std']
config_dict['label_types_train'] += ['pose_mean', 'pose_std']
config_dict['latent_dropout'] = 0

config_dict['swap_appearance'] = False
config_dict['shuffle_3d'] = False
config_dict['actor_subset'] = [1]
config_dict['useCamBatches'] = 0
config_dict['useSubjectBatches'] = 0
config_dict['train_scale_normalized'] = 'mean_std'

# pose training on full dataset
#config_dict['actor_subset'] = [1,5,6,7,8]
