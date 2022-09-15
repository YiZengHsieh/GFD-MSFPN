import h5py
import json
import PIL.Image as Image
import numpy as np
import os
from image import *
# from model_re import CANNet2s
# from model_resnet import CANNet2s
from model_sfa import Model
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import pandas as pd

from torchvision import transforms


# np.set_printoptions(threshold=np.inf)
def plotDensity(density,plot_path,a,typ):
    # print(plot_path)

    num = np.sum(density)
    
    print(num)  

#################結果輸出#################
    # path1 = "result.txt"
    # f = open(path1, 'a')
    # f.write(str(num))
    # f.write("\n")
    # f.close()
    

    '''
    @density: np array of corresponding density map
    @plot_path: path to save the plot
    '''
    density= density*255.0

    #plot with overlay
    colormap_i = cm.jet(density)[:,:,0:3]
    # print(colormap_i)

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:,:,0] = overlay_i[:,:,2]
    new_map[:,:,2] = overlay_i[:,:,0]

    img = new_map*255
    # print(img)
    #text = str(np.ceil(num))
    #cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(plot_path,img)
    # cv2.imwrite("plot/001_test_data/"+ str(a) + typ + ".jpg",img)
    #return (around,around_total)


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# json file contains the test images
test_json_path = './test.json'

# the folder to output density map and flow maps
output_folder = './plot'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)


# model =CANNet2s()
model = Model()

model = model.cuda()

# checkpoint = torch.load('C:/Users/ChiTron/Desktop/thesis/Fish-Flows-main/weight/20220626_re/checkpoint.pth.tar')
# checkpoint = torch.load('C:/Users/ChiTron/Desktop/thesis/Fish-Flows-main/weight/20220628_ResNet18/checkpoint.pth.tar')
checkpoint = torch.load('C:/Users/ChiTron/Desktop/thesis/Fish-Flows-main/weight/20220703_SFANet/model_best.pth.tar')
# checkpoint = torch.load('C:/Users/ChiTron/Desktop/thesis/Fish-Flows-main/model_best.pth.tar')


model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []

a = 1

around = 1
around_total = 0

for i in range(len(img_paths)):
    img_path = img_paths[i]
    print(img_path)

    img_folder = os.path.dirname(img_path)
    # print(img_folder)
    img_name = os.path.basename(img_path)
    # print(img_name)
    index = int(img_name.split('.')[0])

    prev_index = int(max(1,index-1))
    prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))
    print(prev_img_path)

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')

    prev_img = prev_img.resize((640,360))
    img = img.resize((640,360))

    prev_img = transform(prev_img).cuda()
    img = transform(img).cuda()

    # print(img_path)
    gt_path = img_path.replace('.jpg','_resize.h5')
    gt_file = h5py.File(gt_path)
    # print(gt_file['density'])
    target = np.asarray(gt_file['density'])
    # print(target.tolist())
    

    # if i == 0:
    #     print("*********")
    #     path = 'target.txt'
    #     f = open(path, 'w')
    #     f.write(str(target.tolist()))
    #     f.close()


    prev_img = prev_img.cuda()
    prev_img = Variable(prev_img)

    img = img.cuda()
    img = Variable(img)


    img = img.unsqueeze(0)
    prev_img = prev_img.unsqueeze(0)

    prev_flow = model(prev_img,img)


    prev_flow_inverse = model(img,prev_img)

    mask_boundry = torch.zeros(prev_flow.shape[2:])
    mask_boundry[0,:] = 1.0
    mask_boundry[-1,:] = 1.0
    mask_boundry[:,0] = 1.0
    mask_boundry[:,-1] = 1.0

    mask_boundry = Variable(mask_boundry.cuda())

    reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry

    reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry

    overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).data.cpu().numpy()
    

    base_name = os.path.basename(img_path)
    folder_name = os.path.dirname(img_path).split('/')[-1]
    gt_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_gt.jpg')
    density_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_pred.jpg')
    flow_1_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_1.jpg')
    flow_2_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_2.jpg')
    flow_3_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_3.jpg')
    flow_4_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_4.jpg')
    flow_5_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_5.jpg')
    flow_6_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_6.jpg')
    flow_7_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_7.jpg')
    flow_8_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_8.jpg')
    flow_9_path = os.path.join(output_folder,base_name).replace('.jpg','_'+folder_name+'_flow_9.jpg')

    pred = cv2.resize(overall,(overall.shape[1]*8,overall.shape[0]*8),interpolation = cv2.INTER_CUBIC)/64.0
    prev_flow= prev_flow.data.cpu().numpy()[0]
    flow_1 = cv2.resize(prev_flow[0],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_2 = cv2.resize(prev_flow[1],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_3 = cv2.resize(prev_flow[2],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_4 = cv2.resize(prev_flow[3],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_5 = cv2.resize(prev_flow[4],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_6 = cv2.resize(prev_flow[5],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_7 = cv2.resize(prev_flow[6],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_8 = cv2.resize(prev_flow[7],(640,360),interpolation = cv2.INTER_CUBIC)/64.0
    flow_9 = cv2.resize(prev_flow[8],(640,360),interpolation = cv2.INTER_CUBIC)/64.0

    typ = "_pred"
    plotDensity(pred,density_path,a,typ)
    typ = "_gt"
    plotDensity(target,gt_path,a,typ)
    # plotDensity(flow_1,flow_1_path)
    # plotDensity(flow_2,flow_2_path)
    # plotDensity(flow_3,flow_3_path)
    # plotDensity(flow_4,flow_4_path)
    # plotDensity(flow_5,flow_5_path)
    # plotDensity(flow_6,flow_6_path)
    # plotDensity(flow_7,flow_7_path)
    # plotDensity(flow_8,flow_8_path)
    # plotDensity(flow_9,flow_9_path)


    # plotDensity(flow_1,flow_1_path,1)
    # plotDensity(flow_2,flow_2_path,2)
    # plotDensity(flow_3,flow_3_path,3)
    # plotDensity(flow_4,flow_4_path,4)
    # plotDensity(flow_5,flow_5_path,5)
    # plotDensity(flow_6,flow_6_path,6)
    # plotDensity(flow_7,flow_7_path,7)
    # plotDensity(flow_8,flow_8_path,8)
    # plotDensity(flow_9,flow_9_path,9)

    print(a)
    print('\n')
    a = a+1

