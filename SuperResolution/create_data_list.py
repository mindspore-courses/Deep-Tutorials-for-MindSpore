"""process data for mindspore dataset"""

from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['sr data/train2014',
                                     'sr data/val2014'],
                      test_folders=['sr data/BSDS100',
                                    'sr data/Set5',
                                    'sr data/Set14'],
                      min_size=100,
                      output_folder='./')
