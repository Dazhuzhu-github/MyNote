#!/usr/bin/python
#-*- encoding: utf8 -*-

import rospy
import time
import os
from std_msgs.msg import Bool, String


class TestNode:
    def __init__(self):
        rospy.init_node('test_node', anonymous=True)

        self.is_ready_ = False  # 无人机是否已准备完毕
        self.is_takeoff_command_received_ = False  # 是否收到起飞准许信号

        self.readyPub_ = rospy.Publisher('/ready', Bool, queue_size=100)
        self.seenfirePub_ = rospy.Publisher('/seenfire', Bool, queue_size=100)
        self.targetresultPub_ = rospy.Publisher('/target_result', String, queue_size=100)
        self.donePub_ = rospy.Publisher('/done', Bool, queue_size=100)

        self.takeoffSub_ = rospy.Subscriber('/takeoff', Bool, self.takeoffCallback)

        self.testControl()

    # 模拟一次完整的交互流程
    def testControl(self):
        os.system('clear')
        print('模拟交互程序已启动。')
        print('')
        # 假设程序启动至无人机准备完成需要2s
        time.sleep(2)
        # 向/ready发送消息，表示准备完成
        self.is_ready_ = True
        ready_msg = Bool()
        ready_msg.data = 1
        print('无人机准备完成。')
        self.readyPub_.publish(ready_msg)

        # 等待上位机向/takeoff发布起飞准许信号
        while not self.is_takeoff_command_received_:
            pass

        # 假设无人机起飞、穿过着火点需要5s
        time.sleep(5)
        # 向/seenfire发送消息，表示已穿过着火点
        seenfire_msg = Bool()
        seenfire_msg.data = 1
        print('无人机已穿过着火点。')
        self.seenfirePub_.publish(seenfire_msg)

        # 开始搜寻目标，以下模拟了搜寻目标的过程
        targetresult_msg = String()
        # 用1s发现位置1为篮球
        time.sleep(1)
        targetresult_msg.data = '1b'
        print('检测到位置1为篮球。')
        self.targetresultPub_.publish(targetresult_msg)
        # 用1s发现位置4为空
        time.sleep(1)
        targetresult_msg.data = '4e'
        print('检测到位置4为空。')
        self.targetresultPub_.publish(targetresult_msg)
        # 用1s发现位置2为足球
        time.sleep(1)
        targetresult_msg.data = '2f'
        print('检测到位置2为足球。')
        self.targetresultPub_.publish(targetresult_msg)
        # 用1s发现位置5为空
        time.sleep(1)
        targetresult_msg.data = '5e'
        print('检测到位置5为空。')
        self.targetresultPub_.publish(targetresult_msg)
        # 用1s发现位置3为排球
        time.sleep(1)
        targetresult_msg.data = '3v'
        print('检测到位置3为排球。')
        self.targetresultPub_.publish(targetresult_msg)

        # 假设无人机搜寻完目标至降落要3s
        time.sleep(3)
        # 向/done发送消息，表示已完成降落
        done_msg = Bool()
        done_msg.data = 1
        print('无人机已降落。')
        self.donePub_.publish(done_msg)
        # 并假装自己在13s内获得了100分
        time.sleep(2)

    # 接收上位机发送的准许起飞信号
    def takeoffCallback(self, msg):
        if msg.data and self.is_ready_ and not self.is_takeoff_command_received_:
            self.is_takeoff_command_received_ = True
            print('已接收到准许起飞信号。')
        pass


if __name__ == '__main__':
    tn = TestNode()
