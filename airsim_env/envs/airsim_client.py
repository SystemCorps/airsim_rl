
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    # To solve the conflict btw ROS ans cv2
import cv2

import airsim
import numpy as np
import time
from PIL import Image

class AirSimClient():
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
                                        self.init_pose[2], 2.0).join()

        linSpeed = self.calLinSpeed()
        """
        while (linSpeed < 0.1):
            linSpeed = self.calLinSpeed()
            time.sleep(0.03)
        """
        self.client.rotateToYawAsync(self.init_pose[3]).join()
        """
        while (self.client.getMultirotorState().kinematics_estimated.angular_velocity.z_val < 0.1):
            time.sleep(0.01)
        """
        print("Initial Pose!")



        
    def exec_action(self, action):
        # to_msgpack  cannot use np.float32 type
        ac = [0.0]*5
        for i in range(len(action)):
            ac[i] = float(action[i])
        # client.moveByAngleThrottleAsync(pitch, roll, throttle, yaw_rate, duration);
        self.client.moveByAngleThrottleAsync(ac[0], ac[1], ac[2], ac[3], ac[4]).join()
        #time.sleep(action[4])

        #collision = self.client.getMultirotorState().collision.has_collided
        #print(collision)

        #return collision




    def getDroneCam(self, img_size=(84,84,3)):
        return_img = None

        # get png format and convert RGBA to GRAY
        raw_img = self.client.simGetImage("0", airsim.ImageType.Scene)
        png = cv2.imdecode(airsim.string_to_uint8_array(raw_img), cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(png, cv2.COLOR_BGRA2RGB)
        #rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # resize
        img = cv2.resize(rgb_img, (img_size[0], img_size[1]))
        return_img = img.reshape(img_size)
        
        return return_img


    def simReset(self):
        self.client.reset()
        time.sleep(0.2)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # Takeoff
        self.client.takeoffAsync().join()
        self.mvToInitPose()
        print("Simulation Reset Done!")

        
