import os
import h5py
import argparse
from imageretrievalnet import init_network, extract_vectors


parser = argparse.ArgumentParser(description='CNN Image Retrieval Extract Feature')

# options
parser.add_argument('--image-size', '-imsize', default=480, type=int, metavar='N',
                    help='size of longer image side used for extracting feature (default: 480)')
parser.add_argument('--model-choose', '-model', default='vgg16', type=str,
                    help='model for extracting feature (default: vgg16)')
parser.add_argument('--feature-path', '-spath', default='./feature/', type=str,
                    help='path for save feature (default: ./feature/)')
parser.add_argument('--image-path', '-impath', default='./data/', type=str,
                    help='path for image (default: ./data/)')

def get_imlist(path):
    imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist.sort()
    return imlist

def save_feature(images, model_choose, image_size, feature_path, istest):
    
    model = init_network(model=model_choose)
    vecs_MAC, vecs_SPoC, vecs_RMAC, vecs_RAMAC, name_list = extract_vectors(model, images, image_size, print_freq=100)
    feats_MAC = vecs_MAC.numpy()
    feats_SPoC = vecs_SPoC.numpy()
    feats_RMAC = vecs_RMAC.numpy()
    feats_RAMAC = vecs_RAMAC.numpy()

    if os.path.exists(feature_path) == False:
        os.mkdir(feature_path)
    if istest == 0:
        name = feature_path+'feat_'+model_choose+'.h5'
    else:
        name = feature_path+'feat_test_'+model_choose+'.h5'
    h5f = h5py.File(name, 'w')
    h5f.create_dataset('feats_MAC', data = feats_MAC)
    h5f.create_dataset('feats_SPoC', data = feats_SPoC)
    h5f.create_dataset('feats_RMAC', data = feats_RMAC)
    h5f.create_dataset('feats_RAMAC', data = feats_RAMAC)
    h5f.create_dataset('name_list', data = name_list)
    h5f.close()
    print('\r>>>> save to {}.'.format(name))

def main():
    args = parser.parse_args()
    images = get_imlist(args.image_path)
    save_feature(images, args.model_choose, args.image_size, args.feature_path, 0)

if __name__ == '__main__':
    main()
