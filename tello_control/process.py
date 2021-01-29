#!/usr/bin/env python
#coding:utf-8

import rospy
import cv2
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge,CvBridgeError
import detect
import time
targetresultPub = rospy.Publisher('/target_result', String, queue_size=100)
donePub_ = rospy.Publisher('/done', Bool, queue_size=100)
count=0
flag = [0,0,0,0]
seq = [3,1,4,5]

def callback(data):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    result = detect.see(image)
    print(result)
    # here 
    global count , flag ,seq
    count+=1
    idd = seq[count-1]
    name=str(idd)
    if (result[0] == "b"):
        command = name+"b"
	flag[0]=1
    elif (result[0] == "v"):
        command = name+"v"
	flag[1]=1
    elif (result[0]=="f"):
        command = name+"f"
	flag[2]=1
    else:
        command = name+"e"
	if flag[3] == 0 :
	    flag[3]=-1
	else:
	    flag[3]=1
    print(command)
    global targetresultPub ,donePub_
    targetresultPub.publish(command)
    if count == 4 :
	names=["b","v","f","e"]
	last=""
	for i in range (4):
	    if flag[i]!=1:
		last=names[i]
		break
	command = "2"+last
	print(command)
        targetresultPub.publish(command)
	done_msg = Bool()
        done_msg.data = 1
	time.sleep(2)
        print('无人机已降落。')
	
        donePub_.publish(done_msg)

def showImage():
    rospy.init_node('showImage', anonymous = True)
    
    rospy.Subscriber('/ShowImage', Image, callback)
    rospy.spin()
    
if __name__ == '__main__':
    showImage()
