
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    # To solve the conflict btw ROS ans cv2
import cv2

import airsim
import numpy as np
import time

class AirSimClient:
    def __init__(self, init_mean, init_std):
        # Mean and Std of Initial pose of drone - 4-DOF
        self.init_mean = init_mean
        self.init_std = init_std
        
        # AirSim connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # Takeoff
        self.client.takeoffAsync().join()
        
    
    def initPoseRnd(self):
        self.init_pose = np.random.normal(self.init_mean, self.init_std)
    
    
    def calLinSpeed(self):
        state = self.client.getMultirotorState()

        vx = state.kinematics_estimated.linear_velocity.x_val
        vy = state.kinematics_estimated.linear_velocity.y_val
        vz = state.kinematics_estimated.linear_velocity.z_val
        linSpeed = (vx**2 + vy**2 + vz**2)**0.5

        return linSpeed
    
    
    def mvToInitPose(self):
        self.initPoseRnd()
        # To the initial position and yaw (heading)
        self.client.moveToPositionAsync(self.init_pose[0],
                                        self.init_pose[1],
                                        self.init_pose[2])
        self.client.rotateToYawAsync(self.init_pose[3])

        linSpeed = self.calLinSpeed()

        while(linSpeed < 0.1):
            linSpeed = self.calLinSpeed()
            time.sleep(0.03)

        self.ready = True

        
    def exec_action(self, action):
        # client.moveByAngleThrottleAsync(pitch, roll, throttle, yaw_rate, duration);
        self.client.moveByAngleThrottleAsync(action[0], action[1],
                                             action[2], action[3],
                                             action[4])
        

    def getDroneCam(self, img_size):
        return_img = None
        # get png format
        resps = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)])
        resp = resps[0]
        img1d = np.frombuffer(resp.image_data_unit8, dtype=np.uint8)

        
        return return_img
        
