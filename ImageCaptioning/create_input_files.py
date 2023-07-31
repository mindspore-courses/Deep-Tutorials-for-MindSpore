"""Create files for dataset"""

from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='Deep-Tutorials-for-MindSpore/dataset_coco/dataset_coco.json',
                       image_folder='Deep-Tutorials-for-MindSpore/dataset_coco/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='Deep-Tutorials-for-MindSpore/dataset_coco/',
                       max_len=50)
