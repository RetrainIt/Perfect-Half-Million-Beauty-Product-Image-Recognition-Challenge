import os
import h5py
import argparse
import pandas as pd
from extract_feature import save_feature


parser = argparse.ArgumentParser(description='CNN Image Retrieval All Extract Feature')

# options
parser.add_argument('--image-size', '-imsize', default=480, type=int, metavar='N',
                    help='size of longer image side used for extracting feature (default: 480)')
parser.add_argument('--image-path', '-impath', default='./data/', type=str,
                    help='path for image (default: ./data/)')
parser.add_argument('--test-path', '-tpath', default='./val/', type=str,
                    help='path for image (default: ./val/)')
parser.add_argument('--feature-path', '-spath', default='./feature/', type=str,
                    help='path for save feature (default: ./feature/)')
MODEL=['vgg16','resnet101', 'resnext101_64x4d', 'se_resnet101']

def get_imlist(path):
    imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist.sort()
    return imlist

def main():
    args = parser.parse_args()
    # dataset
    images = get_imlist(args.image_path)
    #testdata
    timages = get_imlist(args.test_path)
    
    for model_choose in MODEL:
        save_feature(images, model_choose, args.image_size, args.feature_path, 0)
        save_feature(timages, model_choose, args.image_size, args.feature_path, 1)

if __name__ == '__main__':
    main()
