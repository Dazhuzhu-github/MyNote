#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import detect
import time
import threading
import random
import numpy as np
from collections import deque
from enum import Enum
import os
import rospy
import sys
from std_msgs.msg import Bool, String
#from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# if you can not find cv2 in your python, you can try this. usually happen when you use conda.
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import tello_base as tello

y_max_th = 200
y_min_th = 170

dis=0.0

img = None
tello_state='mid:-1;x:100;y:100;z:-170;mpry:1,180,1;pitch:0;roll:0;yaw:-19;'#位置
#tello_state.find("")
tello_state_lock = threading.Lock()    
img_lock = threading.Lock()    
def sleep(t):
    time.sleep(t)


# send command to tello
class control_handler: 
    def __init__(self, control_pub):
        self.control_pub = control_pub
    
    def forward(self, cm):
        command = "forward "+(str(cm))
        self.control_pub.publish(command)
    
    def back(self, cm):
        command = "back "+(str(cm))
        self.control_pub.publish(command)
    
    def up(self, cm):
        command = "up "+(str(cm))
        self.control_pub.publish(command)
    
    def down(self, cm):
        command = "down "+(str(cm))
        self.control_pub.publish(command)
    
    def right(self, cm):
        command = "right "+(str(cm))
        self.control_pub.publish(command)
    
    def left(self, cm):
        command = "left "+(str(cm))
        self.control_pub.publish(command)

    def cw(self, cm):
        command = "cw "+(str(cm))
        self.control_pub.publish(command)

    def ccw(self, cm):
        command = "ccw "+(str(cm))
        self.control_pub.publish(command)

    def takeoff(self):
        command = "takeoff"
        self.control_pub.publish(command)
        print ("ready")
        
    def mon(self):#motivate
        command = "mon"
        self.control_pub.publish(command)
        print ("mon")

    def land(self):
        command = "land"
        self.control_pub.publish(command)

    def stop(self):
        command = "stop"
        self.control_pub.publish(command)
    def goto(self, strr):
        command = "go "+ strr
        self.control_pub.publish(command)
    def battery(self):
        command = "battery?"
        self.control_pub.publish(command)
#subscribe tello_state and tello_image
class info_updater():   
    def __init__(self):
        rospy.Subscriber("tello_state", String, self.update_state)
        rospy.Subscriber("tello_image", Image, self.update_img)
        self.con_thread = threading.Thread(target = rospy.spin)
        self.con_thread.start()

    def update_state(self,data):
        global tello_state, tello_state_lock
        tello_state_lock.acquire() #thread locker
        tello_state = data.data
        
        tello_state_lock.release()
        # print(tello_state)

    def update_img(self,data):
        global img, img_lock
        img_lock.acquire()#thread locker
        img = CvBridge().imgmsg_to_cv2(data, desired_encoding = "passthrough")
        img_lock.release()
        # print(img)


# put string into dict, easy to find
def parse_state():
    global tello_state, tello_state_lock
    tello_state_lock.acquire()
    statestr = tello_state.split(';')
    print (statestr)
    dict={}
    for item in statestr:
        if 'mid:' in item:
            mid = int(item.split(':')[-1])
            dict['mid'] = mid
        elif 'x:' in item:
            x = int(item.split(':')[-1])
            dict['x'] = x
        elif 'z:' in item:
            z = int(item.split(':')[-1])
            dict['z'] = z
        elif 'mpry:' in item:
            mpry = item.split(':')[-1]
            mpry = mpry.split(',')
            dict['mpry'] = [int(mpry[0]),int(mpry[1]),int(mpry[2])]
        # y can be recognized as mpry, so put y first
        elif 'y:' in item:
            y = int(item.split(':')[-1])
            dict['y'] = y
        elif 'pitch:' in item:
            pitch = int(item.split(':')[-1])
            dict['pitch'] = pitch
        elif 'roll:' in item:
            roll = int(item.split(':')[-1])
            dict['roll'] = roll
        elif 'yaw:' in item:
            yaw = int(item.split(':')[-1])
            dict['yaw'] = yaw
    tello_state_lock.release()
    return dict

def showimg():
    global img, img_lock
    img_lock.acquire()
    #cv2.imshow("tello_image", img)
    #cv2.waitKey(2)
    img_lock.release()
def detectTarget(input):

    image_copy = input.copy()
    height = image_copy.shape[0]
    width = image_copy.shape[1]

    frame = cv2.resize(image_copy, (width, height), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
    frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
    #print(frame.shape)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    #h, s, v = cv2.split(frame)  # 分离出各个HSV通道
    #v = cv2.equalizeHist(v)  # 直方图化
    #frame = cv2.merge((h, s, v))  # 合并三个通道

    #frame = cv2.inRange(frame, color_range_[0], color_range_[1])  # 对原图像和掩模进行位运算
    #print(frame.shape)
    b = frame[:, :, 0].astype(np.int)
    g = frame[:, :, 1].astype(np.int)
    r = frame[:, :, 2].astype(np.int)
    frame=b.astype(np.uint8)
    r1 = r - b
    r2 = r - g
    #cv2.imshow("test",r1.astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.imshow("test", r2.astype(np.uint8))
    #cv2.waitKey(0)
    for i in range(height):
        for j in range(width):
            # dis = (i - 360) ** 2 + (j - 360) ** 2
            if (r1[i, j] > 80 and r2[i, j] > 80):
                frame[i,j]=100
            else :
                frame[i, j] = 0
                # print("yes",i,j,r1[i,j],r2[i,j])




    
    opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
    #print(closed.shape)
    (img ,contours, hierarchy) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

    # 在contours中找出最大轮廓
    contour_area_max = 0
    area_max_contour = None
    #print(contours)
    for c in contours:  # 遍历所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            area_max_contour = c

    isTargetFound = False

    if area_max_contour is not None:
        if contour_area_max > 50:
            isTargetFound = True
    dis=0
    dir=1
    if isTargetFound:
        target = 'Red'
        ((centerX, centerY), rad) = cv2.minEnclosingCircle(area_max_contour)  # 获取最小外接圆
        #print(((centerX, centerY), rad))
        cv2.circle(image_copy, (int(centerX), int(centerY)), int(rad), (0, 255, 0), 2)  # 画出圆心
        dis=(centerX-image_copy.shape[1]/2)
        dis*=(6.5/int(rad))
        print(dis)
        if centerY<image_copy.shape[0]/2 :
            dir=0
        print(int(centerX), int(centerY)), int(rad)
    else:
        target = 'None'
        pass
    cv2.imwrite("sucess.jpg",image_copy)
    if abs(dis) >= 10 and abs(dis) <20 :
        dis=23*(dis/abs(dis))
    

    return dir ,  isTargetFound ,dis
def detect2(input):
    #input=input[:,280:1000,:]



    rxmax = 0
    rxmin = 720
    rymax = 0
    rymin = 960

    b=input[:,:,0].astype(np.int)
    g=input[:,:,1].astype(np.int)
    r=input[:,:,2].astype(np.int)
    r1=r-b
    r2=r-g
    count=0
    print(r2)
    print(input.shape)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
	    #print(i,j)
            #dis = (i - 360) ** 2 + (j - 360) ** 2
            if(r1[i,j]>50 and r2[i,j]>50 ):        
                count+=1
                #print("yes",i,j,r1[i,j],r2[i,j])
                if (i > rxmax):
                    rxmax = i
                if (i < rxmin):
                    rxmin = i
                if (j > rymax):
                    rymax = j
                if (j < rymin):
                    rymin = j

    #ret, thresh1 = cv2.threshold(r1, 127, 255, cv2.THRESH_BINARY)
   # ret2, thresh1 = cv2.threshold(r2, 30, 255, cv2.THRESH_BINARY)

            #print(dis)

                #print("red",i,j)
            #else:
                #input[i,j,:]=[0,0,0]
    cv2.rectangle(input,(rymax,rxmax),(rymin,rxmin),(0,0,225),2)
    result=[(rxmax+rxmin)/2,(rymax+rymin)/2]
    print(result)
    dir = 0
    if (result[0]>360 ):
        dir += 2
    if (result[1 ]>480):
        dir += 1
    flag=0

    if count > 50 : 
	cv2.imwrite("sucess.jpg",input)
        flag=1
    else :
	cv2.imwrite("fail.jpg",input)
    print(dir,flag,count)
    return  dir , flag
# mini task: take off and fly to the center of the blanket.

#result = detect.see(input)



class task_handle():
    class taskstages():  # 大状态
        finding_location  = 0 # find locating blanket
        decision  = 1 # decision
        finished = 6
    class FlightState():  # 移动状态
        WAITING = 2 # 初始状态
        NAVIGATING = 3 # moving
        DETECTING_TARGET = 4  # 拍照？
        LANDING = 5
        HOVER = 7
    # class taskstages(): wlsh5250
    #     finding_location  = 0 # find locating blanket 
    #     order_location  = 1 # find the center of locating blanket and adjust tello 
    #     finished = 6 # task done signal
    #mid 检测挑战卡id，如果未检测到，返回-1
    def __init__(self , ctrl):
        self.States_Dict = parse_state()
        self.ctrl = ctrl
        self.now_stage = self.taskstages.finding_location
        # 旋转参数
        self.R_wu_ = [0, 0, 0, 1]
        self.position = np.zeros([3], dtype=np.float64)
        # decision 用
        self.flight_state_ = self.FlightState.WAITING
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态
        self.window_list_ = [[118,185],[74,135],[118,190],[74,150],[25,185],[-25,135],[25,195],[-25,150]] # 窗户中心点对应的y,z值
        self.win_flag = 0 # 是否穿过窗户 
        self.win_count = 0 # 第几面
        self.dir = 0 #dir的初始状态
        self.ans=["a","a","a","a","a"]
        self.resttime=6 # 控制频率
        self.navigating_queue_ = deque()  # 存放多段导航信息的队列，队列元素为二元list，list的第一个元素代表导航维度（'x' or 'y' or 'z'），第二个元素代表导航目的地在该维度的坐标
        self.navigating_dimension_ = None  # 'x' or 'y' or 'z'
        self.navigating_destination_ = None
        # judge.py used
        #rospy.init_node('test_node', anonymous=True)

        self.is_ready_ = False  # 无人机是否已准备完毕
        self.is_takeoff_command_received_ = False  # 是否收到起飞准许信号
        self.readyPub_ = rospy.Publisher('/ready', Bool, queue_size=100)
        self.seenfirePub_ = rospy.Publisher('/seenfire', Bool, queue_size=100)
        self.impub = rospy.Publisher('/ShowImage', Image, queue_size = 10)
        self.bridge = CvBridge()
        self.targetresultPub_ = rospy.Publisher('/target_result', String, queue_size=100)
        self.donePub_ = rospy.Publisher('/done', Bool, queue_size=100)
        self.takeoffSub_ = rospy.Subscriber('/takeoff', Bool, self.takeoffCallback)
    def takeoffCallback(self, msg):
        if msg.data and self.is_ready_ and not self.is_takeoff_command_received_:
            self.is_takeoff_command_received_ = True
            print('已接收到准许起飞信号。')
        pass
    def main(self): # main function: examine whether tello finish the task
        while not (self.now_stage == self.taskstages.finished):
            if(self.now_stage == self.taskstages.finding_location):
                self.finding_location()
            elif(self.now_stage == self.taskstages.decision):
                print("dec")
                self.decision()
        self.ctrl.land()
        print("Task Done!")
    
    def finding_location(self): # find locating blanket (the higher, the easier)
        assert (self.now_stage == self.taskstages.finding_location)
        while not ( parse_state()['mid'] > 0 ): # if no locating blanket is found:
            distance = random.randint(20,30) # randomly select distance
            print (distance)
            #self.ctrl.battery
            #self.ctrl.up(distance) # tello up ###############################################################################################################################################################3
            #time.sleep(4) # wait for command finished
            showimg()
        print("Find locating blanket!")
        self.now_stage = self.taskstages.decision

    def decision(self):# 移动
        assert (self.now_stage == self.taskstages.decision)
        #rate = rospy.Rate(0.5)
        print("while")
        while not (self.now_stage == self.taskstages.finished):
            if self.flight_state_ == self.FlightState.WAITING:  # 起飞并飞至离墙体（y = 3.0m）适当距离的位置
                # send the ready message to the judge
                self.is_ready_ = True
                ready_msg = Bool()
                ready_msg.data = 1
                print('无人机准备完成。')
                self.readyPub_.publish(ready_msg)

                rospy.logwarn('State: WAITING')
                #self.publishCommand('takeoff')
                #self.navigating_queue_ = deque([['z',155],['y', 0],['x', -230]] ) # right_win ########################################################################
                self.navigating_queue_ = deque([['g',[-230,0,155]]] )
                self.next_state_ = self.FlightState.DETECTING_TARGET  # 初始位置
                self.switchNavigatingState()
                

            elif self.flight_state_ == self.FlightState.NAVIGATING:
                rospy.logwarn('State: NAVIGATING')
                # 如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
                self.States_Dict = parse_state()  # 获取位置
                dist = self.navigating_destination_
                dist_gx = 0.0
                dist_gy = 0.0
                dist_gz = 0.0
                yaw = self.States_Dict['yaw']
                #pitch = self.States_Dict['pitch']
                #roll = self.States_Dict['roll']
                #rospy.logwarn(self.t_wu_)
                #rospy.logwarn(self.des)'
                temp=self.dir
                if self.navigating_dimension_ == 'r':
                    self.dir = self.navigating_destination_

                yaw_diff = (temp - self.dir) 
		#yaw_diff += 65

                if yaw_diff < -180:
                    yaw_diff += 360
                if yaw_diff > 180:
                    yaw_diff -= 360
                #yaw_diff=0
                #if yaw_diff > 15:  # clockwise
                    # yaw_diff= yaw_diff%360
                    #self.ctrl.cw(int(yaw_diff))  # if yaw_diff > 15 else 15
                    #rospy.logwarn(yaw_diff)
                    #sleep(1)
                    #return8
                #elif yaw_diff < -15:  # counterclockwise
                # TODO 1: 发布相应的tello控制命令
                # yaw_diff= - (abs(yaw_diff)%360)
                    #self.ctrl.ccw(int(-yaw_diff))  # if yaw_diff < -15 else 15
                    #rospy.logwarn(yaw_diff)
                    #sleep(1)
                    #return
            # end of TODO 1
                dim_index = 0 if self.navigating_dimension_ == 'x' else (1 if self.navigating_dimension_ == 'y' else 2)
		
		

                
            # pass
                if (self.navigating_dimension_ == 'x'):
                    dist = (self.navigating_destination_ - self.States_Dict['x'])
                elif (self.navigating_dimension_ == 'y'):
                    dist = (self.navigating_destination_ - self.States_Dict['y'])
                elif (self.navigating_dimension_ == 'z'):
                    dist = (self.navigating_destination_ - self.States_Dict['z'])
                elif (self.navigating_dimension_ == 'g'):
                    dist_gx = (self.navigating_destination_[0] - self.States_Dict['x'])
                    dist_gy = (self.navigating_destination_[1] - self.States_Dict['y'])
                    dist_gz = (self.navigating_destination_[2] - self.States_Dict['z'])

                    
                    #dist = math.sqrt(dist_gx**2+dist_gy**2+dist_gz**2) 
                    dist = abs(dist_gx)
                    if (abs(dist_gy) >dist) :
                        dist = abs(dist_gy)
                    if (abs(dist_gz) > dist):
                        dist = abs(dist_gz)
                   

                if self.navigating_dimension_ == 'r':
                    if yaw_diff >0 :
                        yaw_diff=abs(yaw_diff)
                        self.ctrl.cw(int(yaw_diff))
                    else:
                        yaw_diff=abs(yaw_diff)
                        self.ctrl.ccw(int(yaw_diff))
                    rospy.logwarn("turn1")
                    self.switchNavigatingState()
                    rospy.logwarn("turn")
                    global dis
                    dis= yaw_diff/90+2
                elif abs(dist) < 20:  # 当前段导航结束
                    
                    self.switchNavigatingState()
                elif self.navigating_dimension_ == 'g':
                    command_matrix = [[['forward ', 'back '], ['left ', 'right '], ['up ', 'down ']],

                              [['right ', 'left '], ['forward ', 'back '], ['up ', 'down ']],

                              [['back ', 'forward '], ['right ', 'left '], ['up ', 'down ']],

                              [['left ', 'right '], ['back ', 'forward '], ['up ', 'down ']]]

                    dirx_index = 0 if dist_gx > 0 else 1
                    diry_index = 0 if dist_gy > 0 else 1
                    dirz_index = 0 if dist_gz > 0 else 1
                    ind = self.dir / 90
                    command_x = command_matrix[ind][0][dirx_index]
                    command_y = command_matrix[ind][1][diry_index]
                    command_z = command_matrix[ind][2][dirz_index]
                    go_x = 0.0
                    go_y = 0.0
                    go_z = 0.0
                    
                    if (command_z == 'up '):
                        go_z = abs(dist_gz)
                    elif (command_z == 'down '):
                        go_z = -abs(dist_gz)
                    
                    if (command_x == 'right '):
                        go_y = -abs(dist_gx)
                    elif (command_x == 'left '):
                        go_y = abs(dist_gx)
                    elif (command_x == 'forward '):
                        go_x = abs(dist_gx)
                    elif (command_x == 'back '):
                        go_x = -abs(dist_gx)
                    # dealing command y    
                    if (command_y == 'right '):
                        go_y = -abs(dist_gy)
                    elif (command_y == 'left '):
                        go_y = abs(dist_gy)
                    elif (command_y == 'forward '):
                        go_x = abs(dist_gy)
                    elif (command_y == 'back '):
                        go_x = -abs(dist_gy)
                    print(command_x,command_y,command_z,go_x,go_y,go_z)
                    self.ctrl.goto(str(go_x)+" "+str(go_y)+" "+str(go_z)+" "+str(60))
                    global dis
                    dis = math.sqrt(go_x**2+go_y**2+go_z**2)
                    
                    
                
                else:
                    dir_index = 0 if dist > 0 else 1  # direction index
            # TODO 2: 根据维度（dim_index）和导航方向（dir_index）决定使用哪个命令
                    command_matrix = [[['forward ', 'back '], ['left ', 'right '], ['up ', 'down ']],

                              [['right ', 'left '], ['forward ', 'back '], ['up ', 'down ']],

                              [['back ', 'forward '], ['right ', 'left '], ['up ', 'down ']],

                              [['left ', 'right '], ['back ', 'forward '], ['up ', 'down ']]]

                    ind = self.dir / 90
                    command = command_matrix[ind][dim_index][dir_index]                                            
                    print(command,dist)
                    dist=abs(dist)		  		
                    if abs(dist) > 20:
                        if (command == 'up '):
                            print(command,dist)
                            self.ctrl.up(dist)
                        elif (command == 'down '):
                            self.ctrl.down(dist)
                        elif (command == 'right '):
                            self.ctrl.right(dist)
                        elif (command == 'left '):
                            self.ctrl.left(dist)
                        elif (command == 'forward '):
                            self.ctrl.forward(dist)
                        elif (command == 'back '):
                            self.ctrl.back(dist)
                        
                    # print("send")
                    global dis
                    dis=dist
                    # sleep(dist/20+0.5)
                    # self.ctrl.stop
        	# end of TODO 2
                    #TODO!!!!!!!!!!!!!!!!!!!!!!!

            elif self.flight_state_ == self.FlightState.DETECTING_TARGET:
                #time.sleep(3)
                rospy.logwarn('State: DETECTING_TARGET')
                #self.ctrl.forward(50)
                #self.ctrl.forward(dist)
                self.States_Dict = parse_state()
                global img
                yaw = self.States_Dict['yaw']
                #pitch = self.States_Dict['pitch']
                #roll = self.States_Dict['roll']
                yaw_diff = yaw # +65
                yaw_diff=0
                if yaw_diff > 15:  # clockwise
                    self.ctrl.cw(int(yaw_diff) if yaw_diff > 15 else 15)
                    #return
                elif yaw_diff < -15:  # counterclockwise
                    self.ctrl.ccw((int(-yaw_diff) if yaw_diff < -15 else 15))
                    #return
		
                if self.win_flag == 0 :
                    img_lock.acquire()
                    #cv2.imshow("tello_image", img)
                    #cv2.waitKey(2)
                    img2=img.copy()
                    dir ,flag ,dis = detectTarget(img2)
                    print(dis)
                    img_lock.release()
                    if flag:
                        rospy.loginfo('Target detected.')
                        print(dir)
                        #if self.win_count == 0:
                          #  dir+=4
                        #win_count+=1
                        dis=int(dis)
                        self.ctrl.goto(str(20)+" "+str(-dis)+" "+str(0)+" "+str(60))
                        #if dis >20 :
                            #self.ctrl.right(abs(dis))
                         #   self.ctrl.goto(str(20)+" "+str(-dis)+" "+str(0)+" "+str(60))
                        #elif dis < -20 :
                            #self.ctrl.left(abs(dis))
                         #   self.ctrl.goto(str(20)+" "+str(-dis)+" "+str(0)+" "+str(60))
                        self.navigating_queue_ = deque([['z',self.window_list_[dir][1]],['x',-100]])
                        self.flight_state_=self.FlightState.NAVIGATING 
                        self.next_state_ = self.FlightState.DETECTING_TARGET  
                        self.switchNavigatingState()
                        self.win_flag =1
                                #ctrl.land()
                    else :
                        self.win_count +=1
                        if self.win_count == 1:
                    
                            rospy.loginfo('Target not  detected.')
                            #self.navigating_queue_ = deque([['y', 95], ['x', -230],['z',155]])
                            self.navigating_queue_ = deque([['g',[-230,95,155]]])
                            self.flight_state_=self.FlightState.NAVIGATING 
                            self.next_state_ = self.FlightState.DETECTING_TARGET  
                            self.switchNavigatingState()
                    #else :
                    # self.ctrl.land()
                else:
                    # 向/seenfire发送消息，表示已穿过着火点
                    seenfire_msg = Bool()
                    navl=[]
                    # 1
                    #navl.append(['x',-70])
                    # navl.append(['y',-70])
                    # navl.append(['x',60])
                    # navl.append(['y',-70])
                    #navl.append(['g',[60,-70,150]])
                    #navl.append(['z',150])
                    navl.append(['r',180])
                    navl.append(['g',[15,-70,100]])
                    navl.append(['h',1])
                    # 2
                    navl.append(['y',-40])
                    # navl.append(['x',10])
                    # navl.append(['z',110])
                    navl.append(['r',90])
                   
                    navl.append(['h',2])
                    
                    # 3
                    #navl.append(['r',90])
                    navl.append(['g',[70,-80,150]])
                    #navl.append(['z',130])
                    navl.append(['h',3])
                    # 4
                    navl.append(['r',270])
                    # navl.append(['x',140])
                    # navl.append(['y',70])
                    # navl.append(['z',130])
                    navl.append(['g',[140,-60,130]])
                    navl.append(['h',4])
                    # land
                    # navl.append(['x',210])
                    # navl.append(['y',-10])
                    navl.append(['g',[200,-60,70]])
                    self.navigating_queue_ = deque(navl)
                    self.switchNavigatingState()
                    self.next_state_ = self.FlightState.LANDING

                    #self.ctrl.land()





            elif self.flight_state_ == self.FlightState.LANDING:
                rospy.logwarn('State: LANDING')
                # 向/done发送消息，表示已完成降落
                done_msg = Bool()
                done_msg.data = 1
                print('无人机已降落。')
                self.donePub_.publish(done_msg)
                self.now_stage = self.taskstages.finished
                # for路径1，补全没检测的3号位

                self.ctrl.land()#('land')

            elif self.flight_state_ == self.FlightState.HOVER:
                rospy.logwarn('State: hover')
                targetresult_msg = String()
                # 用1s发现位置1为篮球
                name=str(self.navigating_destination_)
                if  name == "2":
                    sleep(2)
                #sleep(1)
                img_lock.acquire()
                    #cv2.imshow("tello_image", img)
                    #cv2.waitKey(2)
                img2=img.copy()
                command = "1b"
                img_lock.release()
                cv2.imwrite((name+".jpg"),img2)
                #result = detect.see(img2)
                self.impub.publish(self.bridge.cv2_to_imgmsg(img2, "bgr8"))
                result="a"
                if result == None :
                    result = "a"
                print(result)
                if (result[0] == "b"):
                    command = name+"b"
                elif (result[0] == "v"):
                    command = name+"v"
                elif (result[0]=="f"):
                    command = name+"f"
                else:
                    command = name+"e"
                self.targetresultPub_.publish(command)
                # targetresult_msg.data = '1b'
                # #print('检测到位置1为篮球。')
                # self.targetresultPub_.publish(targetresult_msg)
                # # 用1s发现位置4为空
                # #time.sleep(1)
                # targetresult_msg.data = '4e'
                # #print('检测到位置4为空。')
                # self.targetresultPub_.publish(targetresult_msg)
                # # 用1s发现位置2为足球
                # #time.sleep(1)
                # targetresult_msg.data = '2f'
                # #print('检测到位置2为足球。')
                # self.targetresultPub_.publish(targetresult_msg)
                # # 用1s发现位置5为空
                # #time.sleep(1)
                # targetresult_msg.data = '5e'
                # #print('检测到位置5为空。')
                # self.targetresultPub_.publish(targetresult_msg)
                # # 用1s发现位置3为排球
                # #time.sleep(1)
                # targetresult_msg.data = '3v'
                # print('检测到位置3为排球。')
                # self.targetresultPub_.publish(targetresult_msg)
                self.switchNavigatingState()

            else:
                pass

            #rate.sleep()
            global dis
            print("sleep")
            print(dis)

            if dis > 0 :
                if dis > 50 :
                    sleep(dis/15)
                else:
                    sleep(dis/6)
            dis=0
            self.ctrl.stop

        #self.switchNavigatingState()    
    def switchNavigatingState(self ):
        if len(self.navigating_queue_) == 0:
            self.flight_state_ = self.next_state_
        else: # 从队列头部取出无人机下一次导航的状态信息

            next_nav = self.navigating_queue_.popleft()
	    self.des=next_nav
            self.navigating_dimension_ = next_nav[0]
            self.navigating_destination_ = next_nav[1]
           
            if (self.navigating_dimension_  == 'l'):
                self.flight_state_ = self.FlightState.LANDING
            elif (self.navigating_dimension_ == 'h'):
                self.flight_state_ = self.FlightState.HOVER
            else:
                self.flight_state_ = self.FlightState.NAVIGATING
        #self.goto()




if __name__ == '__main__':
    rospy.init_node('tello_control', anonymous=True)

    control_pub = rospy.Publisher('command', String, queue_size=1)
    ctrl = control_handler(control_pub)
    infouper = info_updater()
    tasker = task_handle(ctrl)
    
    sleep(2)
    ctrl.mon()
    sleep(2.5)
    ctrl.takeoff()
    sleep(1.5)
    #ctrl.up(30)
    print("up 30")
    #time.sleep(2)
    #time.sleep(5)
    # while(1):
    #     if parse_state()['mid'] == -1:
    #         ctrl.takeoff()
    #         print("take off")
    #         break
    #print("mon")
    #ctrl.forward(20)
    #time.sleep(4)
    #ctrl.up(50)
    #print("up 60")
    #time.sleep(2)
    

    #time.sleep(2)

    tasker.main()
    # for i in range(0):
	# #print(tello_state)
	# ctrl.up(20)
	# time.sleep(2)
	# ctrl.left(20)
	# time.sleep(2)
	# ctrl.down(20)
	# time.sleep(2)
	# ctrl.right(20)
	# time.sleep(2)
	# print(tello_state)
    ctrl.land()

    

