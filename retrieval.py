import os
import h5py
import copy
import faiss
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from extract_feature import save_feature

parser = argparse.ArgumentParser(description='Image Retrieval')

# options
parser.add_argument('--image-size', '-imsize', default=480, type=int, metavar='N',
                    help='size of longer image side used for extracting feature (default: 480)')
parser.add_argument('--is-eval', '-iseval', default=1, type=int, metavar='N',
                    help='is eval (default: 1)')
parser.add_argument('--image-path', '-impath', default='./data/', type=str,
                    help='path for image (default: ./data/)')
parser.add_argument('--test-path', '-tpath', default='./val/', type=str,
                    help='path for image (default: ./val/)')
parser.add_argument('--test-label', '-tlabel', default='./val.csv', type=str,
                    help='label for test image (default: ./val.csv)')
parser.add_argument('--feature-path', '-spath', default='./feature/', type=str,
                    help='path for save feature (default: ./feature/)')
MODEL=['vgg16','resnet101', 'resnext101_64x4d', 'se_resnet101']

def get_imlist(path):
    imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    imlist.sort()
    return imlist

def read_feature(name):
    h5f = h5py.File(name,'r')
    feats_MAC = h5f['feats_MAC'][:]
    feats_SPoC = h5f['feats_SPoC'][:]
    feats_RMAC = h5f['feats_RMAC'][:]
    feats_RAMAC = h5f['feats_RAMAC'][:]
    name_list = h5f['name_list'][:]
    h5f.close()
    return feats_MAC, feats_SPoC, feats_RMAC, feats_RAMAC, name_list

def main():
    args = parser.parse_args()
    
    # read database
    name = args.feature_path+'feat_'+MODEL[0]+'.h5'
    vgg_feats_MAC, vgg_feats_SPoC, vgg_feats_RMAC, vgg_feats_RAMAC, data_list = read_feature(name)
    name = args.feature_path+'feat_'+MODEL[1]+'.h5'
    resnet_feats_MAC, resnet_feats_SPoC, resnet_feats_RMAC, resnet_feats_RAMAC, _ = read_feature(name)
    name = args.feature_path+'feat_'+MODEL[2]+'.h5'
    resnext_feats_MAC, resnext_feats_SPoC, resnext_feats_RMAC, resnext_feats_RAMAC, _ = read_feature(name)
    name = args.feature_path+'feat_'+MODEL[3]+'.h5'
    senet_feats_MAC, senet_feats_SPoC, senet_feats_RMAC, senet_feats_RAMAC, _ = read_feature(name)
    
    # read test
    name = args.feature_path+'feat_test_'+MODEL[0]+'.h5'
    vgg_feats_test_MAC, vgg_feats_test_SPoC, vgg_feats_test_RMAC, vgg_feats_test_RAMAC, test_list = read_feature(name)
    name = args.feature_path+'feat_test_'+MODEL[1]+'.h5'
    resnet_feats_test_MAC, resnet_feats_test_SPoC, resnet_feats_test_RMAC, resnet_feats_test_RAMAC, _ = read_feature(name)
    name = args.feature_path+'feat_test_'+MODEL[2]+'.h5'
    resnext_feats_test_MAC, resnext_feats_test_SPoC, resnext_feats_test_RMAC, resnext_feats_test_RAMAC, _ = read_feature(name)
    name = args.feature_path+'feat_test_'+MODEL[3]+'.h5'
    senet_feats_test_MAC, senet_feats_test_SPoC, senet_feats_test_RMAC, senet_feats_test_RAMAC, _ = read_feature(name)
    
    feats=np.concatenate((vgg_feats_MAC,vgg_feats_SPoC,vgg_feats_RMAC,vgg_feats_RAMAC,\
                       resnet_feats_MAC, resnet_feats_SPoC, resnet_feats_RMAC, resnet_feats_RAMAC,\
                       resnext_feats_MAC, resnext_feats_SPoC, resnext_feats_RMAC, resnext_feats_RAMAC,\
                       senet_feats_MAC, senet_feats_SPoC, senet_feats_RMAC, senet_feats_RAMAC),axis=1)

    feats_test=np.concatenate((vgg_feats_test_MAC, vgg_feats_test_SPoC, vgg_feats_test_RMAC, vgg_feats_test_RAMAC,\
                        resnet_feats_test_MAC, resnet_feats_test_SPoC, resnet_feats_test_RMAC, resnet_feats_test_RAMAC,\
                        resnext_feats_test_MAC, resnext_feats_test_SPoC, resnext_feats_test_RMAC, resnext_feats_test_RAMAC,\
                        senet_feats_test_MAC, senet_feats_test_SPoC, senet_feats_test_RMAC, senet_feats_test_RAMAC),axis=1)
                        
    # z-normalization
    feats = (feats - np.mean(feats, axis=0)) / np.std(feats, axis=0)
    feats_test = (feats_test - np.mean(feats_test, axis=0)) / np.std(feats_test, axis=0)
    
    # PCA
    print(feats.shape)
    print(feats_test.shape)
    pca = PCA(n_components=2048,svd_solver='full', random_state=2018)
    pca.fit(feats)
    feats=pca.transform(feats)
    feats_test=pca.transform(feats_test)
    
    #DBA
    DBA_num = 2
    #res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(feats.shape[1])
    #gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
    feats=feats.astype('float32')
    index_flat.add(feats)
    D,I = index_flat.search(feats,DBA_num)
    
    new_feats = copy.deepcopy(feats)
    for num in range(len(I)):
        new_feat = feats[I[num][0]]
        for num1 in range(1,len(I[num])):
            weight = (len(I[num])-num1) / float(len(I[num]))
            new_feat += feats[num1] * weight
        new_feats[num]=new_feat

    #QE
    QE_num = 2
    #res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(new_feats.shape[1])
    #gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
    feats_test=feats_test.astype('float32')
    index_flat.add(new_feats)
    D,I = index_flat.search(feats_test,QE_num - 1)
    query_feat=copy.deepcopy(feats_test)
    for num in range(len(query_feat)):
        query_feat[num]=(query_feat[num]+new_feats[I.T[0,num]]) / float(QE_num)

    # final query
    #res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(new_feats.shape[1])
    #gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
    index_flat.add(new_feats)
    D,I = index_flat.search(query_feat,7)
    result_query = I.T

    # save result
    pd.DataFrame(result_query).to_csv('./result_query.csv')

    #eval
    if args.is_eval == 1:
        # read label
        val=pd.read_table(args.test_label)
        val.columns=[1]
        query_list = list(val[1].map(lambda x: args.test_path + x.split(',')[0] + '.jpg'))
        val_label = []
        for num in range(len(val[1])):
            t=[]
            for num1 in range(len(val[1][num].split(','))-1):
                t.append(val[1][num].split(',')[num1+1]+'.jpg')
            val_label.append(t)

        # caculate MAP
        one = 0
        apt=0
        tt=0
        ap=0
        for query_num in range(len(val_label)):
            queryDir = query_list[query_num]
            top_num = 7
            imlist = [data_list[index] for i,index in enumerate(result_query[0:top_num,query_num])]
            for num in range(top_num):
                if imlist[num] in set(val_label[query_num]):
                    one += 1
                    tt += one/float(num+1)
            if one!= 0:
                ap = tt/one
                apt += ap
            tt=0
            one=0
        MAP_score = apt/100
        print('\r>>>> MAP@7 is {}.'.format(MAP_score))

if __name__ == '__main__':
  main()
