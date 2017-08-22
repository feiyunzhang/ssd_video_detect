#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pylab
import imageio
import skimage.io
import numpy as np  
import cv2  
cv2.__version__
import sys
import os
import time
#import cv2.cv as cv
#from cv2 import cv
import Queue
import multiprocessing



cap = cv2.VideoCapture('/disk/fyzhang/ssd/caffe-ssd/video/car.mp4')  
#fps = cap.get(cv2.CAP_PROP_POS_FRAMES)
fps = cap.get(cv2.CAP_PROP_FPS)
total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
width=size[0]
height=size[1]
frame_count = 0
all_frames = []
#print total
#print fps,size
#print width
arr=[]
def countfp(array):
	fp_object_count=0
	if len(array)>2:
		array = array[-2:]
        
		if (array[1]==arr[1] and array[0]!=0):
			fp_object_count=fp_object_count+1
		else:
			return fp_object_count		
			fp_object_count=0
while(cap.isOpened()):  
    ret, frame = cap.read()  
    all_frames.append(frame)
    #print all_frames
    #print frame
    #os.system("pause")
    #time.sleep(20)
    frame_count = frame_count + 1
    
    q=Queue.Queue(2)
    try:
    #q.put(1)
    	q.put(frame_count)
	q.put(frame_count)
	#print(q.qsize())
    	first=q.get()
    	#second=q.get()
    	#print first#,second
	#print(q.qsize())
    except queue.Empty as q_error:
    	print('The Queue is empty!')
    arr.append(frame_count)
    #print arr
    #print arr[frame_count-1]
    countfp(arr)
    
    #L.remove(var)
    
    if (ret ):
	try:
		f=open("a.json","w")
	#for i in range(1,10):
        #json=str({\"framerate\": "%f" %fps,\"width\": "%d" %width})
        	print >>f,  '{'+'"framerate"'':''"%f"'%fps,',''"totaltime"'':''"%f"'%totialtime,',''"width"'':''"%d"' %width +'}'
	
        finally:
		if f:
	#f.write('"framerate"'':' '"%f"' %fps,'"width"'':' '"%d"' %width)
        #f.write('{'+'"framerate"'':' '"%f"' %fps,'"width"'':' '"%d"' %width +'}')
	
			f.close()
    	cv2.imshow('image', frame) 
    else:
	break
    #print frame_count
    #print len(all_frames)   
    

    k = cv2.waitKey(1)  
    if (k & 0xff == ord('q')):  
        break  
cap.release()  
cv2.destroyAllWindows()
totaltime=frame_count/fps
print totaltime
