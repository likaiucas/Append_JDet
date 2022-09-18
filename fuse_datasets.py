import os
import cv2
import numpy as np
from jdet.data.devkits.convert_data_to_mmdet import convert_data_to_mmdet
from jdet.data.devkits.ImgSplit_multi_process import process
from tqdm import tqdm

root = '/data/preprocessed'
dataset1 = f'{root}/FAIR1M/trainval'
dataset2 = f'{root}/mid/train'
fusion = f'{root}/fusion'
new_train = f'{root}/new_training'

FAIR_minPixel1 = {'Airplane':100000000, 'Ship':2000000000, 'Vehicle':100000000, 'Basketball_Court':0, 'Tennis_Court':0, 
        "Football_Field":0, "Baseball_Field":0, 'Intersection':0, 'Roundabout':0, 'Bridge':0}
FAIR_minPixel2 = {'Airplane':0, 'Ship':0, 'Vehicle':0, 'Basketball_Court':0, 'Tennis_Court':0, 
        "Football_Field":0, "Baseball_Field":0, 'Intersection':0, 'Roundabout':0, 'Bridge':0}

assert os.path.exists(dataset1) 
assert os.path.exists(dataset2) 
if not os.path.exists(fusion):
    os.mkdir(fusion)
if not os.path.exists(os.path.join(fusion,'images')):
    os.mkdir(os.path.join(fusion,'images'))
if not os.path.exists(os.path.join(fusion,'labelTxt')):
    os.mkdir(os.path.join(fusion,'labelTxt'))

def return_image_path(dataset,f):
    return os.path.join(dataset,'images',f.replace('.txt', '.png'))

def run(dataset, FAIR_minPixel):
    new_images = os.path.join(fusion,'images')
    new_txt = os.path.join(fusion,'labelTxt')

    files= os.listdir(os.path.join(dataset, 'labelTxt'))
    for f in tqdm(files):
        datas = open(os.path.join(dataset, 'labelTxt',f)).readlines()
        blank_list=[] #记录符合筛选面积的图像
        for data in datas:
            ds = data.split(" ")
            if len(ds) < 10:
                continue
            label = ds[8]
            bbox=[int(float(i)) for i in ds[:8]]
            rect = cv2.minAreaRect(np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]],]),)
            s = rect[1][0]*rect[1][0]
            if FAIR_minPixel[label]<s:
                blank_list.append(data)
        
        if blank_list:
            img_path = return_image_path(dataset,f) #原始图像地址
            if os.path.exists(os.path.join(new_txt,f)):
                f='p'+f
                newim = os.path.join(new_images, f.replace('.txt', '.png'))
                os.system(f'cp {img_path} {newim}')
                with open(os.path.join(new_txt, f), 'w') as txt:
                    for line in blank_list:
                        txt.write(line)
            else:
                os.system(f'cp {img_path} {new_images}')
                with open(os.path.join(new_txt, f), 'w') as txt:
                    for line in blank_list:
                        txt.write(line)                

    
if __name__=='__main__':
    run(dataset1, FAIR_minPixel1)
    run(dataset2, FAIR_minPixel2)
    target_path = process(fusion, new_train, subsize=1024, gap=200, rates=[1.])
    convert_data_to_mmdet(target_path, os.path.join(target_path, 'labels.pkl'), type='FAIR1M_1_5')
