import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import skimage
import skimage.io as skio

import os
from os import path

import warnings
warnings.simplefilter("ignore")

import time

import cv2

COLORS = ((0,0,0), (51, 51, 255), (255, 51, 51), (51, 255, 51), (255,255,0), (0,255,255), (0,127,255), (128,0,255), (102,102,255), (255,102,102), (102,255,102) )


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '.'  # this file is expected to be in {caffe_root}/examples
import os
#os.chdir(caffe_root)
import sys
sys.path.insert(0, caffe_root + "/caffe/python")

from inspect import getmembers, isfunction

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()



from google.protobuf import text_format
from caffe.proto import caffe_pb2 as cpb2

#print cpb2

# load PASCAL VOC labels

voc_labelmap_file = "data/VOC0712/labelmap_voc.prototxt"
file = open(voc_labelmap_file, 'r')

voc_labelmap = cpb2.LabelMap()


text_format.Merge(str(file.read()), voc_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    classindex = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                classindex.append(labelmap.item[i].label)
                break
        assert found == True
    return labelnames, classindex

model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_100000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                1)     # use test mode (e.g., don't perform dropout)

#caffe.TEST

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)



'''
def detect_this_image(image):
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

   # transformed_image=transformer.preprocess('data', image)
   # net.blob['data'].data[0,...]=transformed_image
    detections=net.forward()['detection_out']
    det_label=detections[0,0,:,1]
    det_conf=detections[0,0,:,2]
    det_xmin=detections[0,0,:,3]
    det_ymin=detections[0,0,:,4]
    det_xmax=detections[0,0,:,5]
    det_ymax=detections[0,0,:,6]
    top_indices=[i for i, conf in enumerate(det_conf) if conf >= 0.7]

    top_conf=det_conf[top_indices]
    top_label_indices=det_label[top_indices].tolist()
    top_labels, top_class_index = get_labelname(labelmap, top_label_indices)
    #top_labels=get_labelname(labelmap,top_label_indices)
    top_xmin=det_xmin[top_indices]
    top_ymin=det_ymin[top_indices]
    top_xmax=det_xmax[top_indices]
    top_ymax=det_ymax[top_indices]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    dets=np.empty((0,5),int)
    cname= []

    for i in xrange(top_conf.shape[0]):
	xmin = int(round(top_xmin[i]*image.shape[1]))
        ymin = int(round(top_ymin[i]*image.shape[0]))
        xmax = int(round(top_xmax[i]*image.shape[1]))
        ymax = int(round(top_ymax[i]*image.shape[0]))
        score = top_conf[i]
        label = top_class_index[i]
        label_name = top_labels[i]
        color = colors[label]
        cname = '%s: %.2f'%(label_name, score)
	dets = np.append(dets,np.array([[xmin,ymin,xmax,ymax,label]]),axis=0)
	#dets = []
	#dets = dets.append(np.array([[xmin,ymin,xmax,ymax]])
	#dets = np.append(dets,np.array([[xmin,ymin,xmax,ymax]]))
	#dets = dets.append(name)
    return dets,cname
'''
def predict(image):
#def predict(imgpath, outdir):    
    #start_time = time.time()
    #imagename = imgpath.split('/')[-1]

    #image = cv2.imread(imgpath)
    #cpimg = image.copy()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = skimage.img_as_float(image).astype(np.float32)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.7]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels, top_class_index = get_labelname(voc_labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]


    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    dets=np.empty((0,4),int)
    cname= []


       # score = top_conf[i]
       # label = top_class_index[i]
       # label_name = top_labels[i]
      #  color = colors[label]
      #  cname = '%s: %.2f'%(label_name, score)
	#dets = np.append(dets,np.array([[xmin,ymin,xmax,ymax,label]]),axis=0)
	#dets = []
	#dets = dets.append(np.array([[xmin,ymin,xmax,ymax]])
	#dets = np.append(dets,np.array([[xmin,ymin,xmax,ymax]]))
	#dets = dets.append(name)


    if top_conf.shape[0] > 0:
        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = top_labels[i]
            color_index = top_class_index[i]
            cname = '%s: %.2f'%(label, score)
	    dets = np.append(dets,np.array([[xmin,ymin,xmax,ymax]]),axis=0)
            #if label != "sky" and label != "road":
            #cv2.rectangle(cpimg, (xmin, ymin), (xmax, ymax), COLORS[color_index], 2)
            #cv2.putText(cpimg, name, (xmin, ymin + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[color_index] , 1)

	return dets,cname
        #output_img = path.join(outdir,imagename)

        #cv2.imwrite(output_img,cpimg)
    else:
	for i in xrange(top_conf.shape[0]):
            xmin = 0
            ymin = 0
            xmax = 0
            ymax = 0
            score = 0
            label = 0
            color_index = 0
            cname = 'nothing'
	    dets = np.append(dets,np.array([[xmin,ymin,xmax,ymax,label]]),axis=0)
        #output_img = path.join(outdir ,imagename)
        #print 'bug'
	return dets,cname
        #cv2.imwrite(output_img,cpimg)

    #end_time = time.time()

    #exec_time = end_time - start_time

    #print 'Detect %s in %s seconds' % (imagename, exec_time)


cap = cv2.VideoCapture('/disk/fyzhang/ssd/caffe-ssd/video/hls02.mp4')
while(cap.isOpened()):
    ret,frame = cap.read()
    

#caffe.TEST


    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    image_resize=300
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    #image = cv2.resize(frame,(int(300),int(300)),interpolation=cv2.INTER_CUBIC)
    #image = frame
    #if cv2.waitkey(1) & 0xFF == ord('q'):
    #	break
    k = cv2.waitKey(1)  
    if (k & 0xff == ord('q')):  
        break 
    detss, dnames = predict(frame)
    #if detss 
    #print detss,dnames,len(detss)
    #print dnames
    if len(detss)==1:
	for d in detss:
    		fheight,fwidth = frame.shape[:2]
		xxmin=int((d[0]))
    		yymin=int(d[1])
    		xxmax=int((d[2]))
    		yymax=int(d[3])
		#print int(d[0]),xxmin,int(d[1]),yymin,int(d[3]),xxmax
    		#xxmin=int(round(d[0]*fwidth/300))
    		#yymin=int(round(d[1]*fheight/300))
    		#xxmax=int(round(d[2]*fwidth/300))
    		#yymax=int(round(d[3]*fheight/300))
		#print xxmin,yymin,xxmax,yymax
        	#d = d.astype(np.uint32)
        	color=COLORS[10]
		cv2.rectangle(frame,(xxmin, yymin), (xxmax, yymax), COLORS[10], 2)
		cv2.putText(frame,dnames,(xxmin,yymin),cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[1] , 1)
		cv2.imshow('frame', frame)
    else:
		cv2.imshow('frame', frame)
cap.release()
cv2.destroyAllWindows()




