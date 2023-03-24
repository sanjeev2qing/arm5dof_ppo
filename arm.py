import numpy as np
import torch
import rospy
from arm5dof_interface.srv import GetJointAngle
from arm5dof_interface.action import JointAngleControl, GripperControl
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile
from rclpy.node import Node
from rclpy.action import ActionClient
from ros2_gym.envs import ROS2Env
from ros2_gym.utils import get_tf2_transform


class Arm5DOFEnv(ROS2Env):
    def __init__(self):
        super().__init__('arm5dof_rl')

        # Parameters
        self._goal = np.array([2.0, 2.0, 0.0])
        self._max_episode_steps = 1000
        self._episode_step = 0

        # Services
        self._get_joints_angle_client = self.create_client(GetJointAngle, '/get_joints_angle')
        while not self._get_joints_angle_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self._set_joints_angle_client = self.create_client(JointAngleControl, '/joint_angle_control')
        while not self._set_joints_angle_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self._set_gripper_angle_client = self.create_client(GripperControl, '/gripper_control')
        while not self._set_gripper_angle_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Action clients
        self._move_base_client = ActionClient(self, PoseStamped, '/move_base')
        self._odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, QoSProfile(depth=10))

        # State space
        self.observation_space = np.zeros(5)

        # Action space
        self.action_space = np.zeros(6)

    def reset(self):
        # Reset episode step counter
        self._episode_step = 0

        # Reset robot joints
        joints_angle = self.get_joints_angle()
        joints_angle[-1] = 0.0  # Set gripper angle to 0.0
        self.set_joints_angle(joints_angle)

        # Generate new goal position
        self._goal = np.array([2.0, 2.0, 0.0])

        # Return initial observation
        return self.get_observation()

    def step(self, action):
        # Set robot joints
        joints_angle = np.array(action[:5])
        gripper_angle = action[5]
        joints_angle = np.clip(joints_angle, -1.57, 1.57)  # Clip joint angles to limits
        gripper_angle = np.clip(gripper_angle, 0.0, 1.75)  # Clip gripper angle to limits
        joints_angle = np.append(joints_angle, gripper_angle)
        self.set_joints_angle(joints_angle)

        # Wait for robot to reach new joint angles
        while not self.is_joints_angle_reached(joints_angle):
            self.rate.sleep()

        # Compute reward
        gripper_pos = self.get_gripper_pos()
        distance = np.linalg.norm(gripper_pos - self._goal)
        reward = -distance

        # Check if episode is done
        done = False
        self._episode_step += 1
        if self._episode_step >= self._max_episode_steps:
            done = True

        # Return observation, reward, done, and info
        observation = self.get_observation()
        info = {}
        return observation, reward, done, info

    def get_observation(self):
        # Get robot joints angle
        joints_angle = self.get_joints_angle()

        # Return joints angle as observation
        return joints_angle

    def get_joints_angle(self):
        # Call get_joints_angle service
        request = GetJointAngle.Request()
        request.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'left_gripper_joint']
        future = self._get_joints_angle_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        # Convert response to numpy array
        joints_angle = np.array(response.joint_angle[:-1])

        return joints_angle

    def set_joints_angle(self, joints_angle):
        # Call set_joints_angle service
        request = JointAngleControl.Request()
        request.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        request.joint_angle = list(joints_angle[:-1])
        request.period = 1.0
        request.enable_minimum_jerk = True
        future = self._set_joints_angle_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def is_joints_angle_reached(self, joints_angle, tolerance=0.01):
        # Check if current joints angle is within tolerance to target joints angle
        current_joints_angle = self.get_joints_angle()
        distance = np.linalg.norm(joints_angle - current_joints_angle)
        return distance < tolerance

    def get_gripper_pos(self):
        # Get robot base pose
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = 0.0
        goal.pose.position.y = 0.0
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0
        goal_handle = self._move_base_client.send_goal(goal)

        # Wait for robot to reach base pose
        while not goal_handle.done():
            self.rate.sleep()

        # Get robot end effector pose
        gripper_pos = np.array([0.0, 0.0, 0.0])
        try:
            odom_msg = self._odom_queue.get_nowait()
            quat = (odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                    odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w)
            rot_mat = tf2_geometry_msgs.transformations.quaternion_matrix(quat)
            gripper_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y,
                                    odom_msg.pose.pose.position.z, 1.0])
            gripper_pos = np.matmul(rot_mat, gripper_pos)[:3]
        except Empty:
            pass

        return gripper_pos

    def odom_callback(self, msg):
        self._odom_queue.put(msg)
