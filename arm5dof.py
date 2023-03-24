import time
import os
import sys
import numpy as np
import threading
import pybullet

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.action import ActionServer
import tf2_ros
# Ament
from ament_index_python.packages import get_package_share_directory
# 消息
from sensor_msgs.msg import JointState 
from visualization_msgs.msg import Marker
# 导入自定义库
cur_pkg_path = get_package_share_directory("arm5dof_hardware")
arm5dof_uservo_folder = os.path.join(cur_pkg_path, "arm5dof_python_uservo")
sys.path.append(arm5dof_uservo_folder)
from transform import Transform
from pose import Pose
from arm5dof_uservo import Arm5DoFUServo
from arm_application import ArmApplication
from config import *
# 自定义消息
from arm5dof_interface.msg import JointAngleList, ToolPose
# 自定义服务
from arm5dof_interface.srv import GetArmState, GetJointAngle,\
	GetToolPose, SetDamping
# 自定义动作
from arm5dof_interface.action import GripperControl, \
	JointAngleControl, SetToolPose, MoveToWorkspace, GrabObject

class Arm5DoFUServoServer(Node):
	'''五自由度机械臂PyBullet服务端'''
	# RVIZ2 Marker的ID
	MARKER_ID_CUBIC = 0
	MARKER_WS_01 = 1
	MARKER_WS_02 = 2

	# 轨迹执行回调函数执行周期, 几个时间块执行一次
	trajectory_callback_period = 40
	
	def __init__(self):
		# 父类初始化
		super().__init__("arm5dof_server")
		self._logger.info("开始初始化五自由度机械臂服务器")
		# 机械臂应用场景APP
		config_folder = os.path.join(cur_pkg_path, "arm5dof_python_uservo", "config")
        # 创建仿真机械臂对象
		self.arm = Arm5DoFUServo(config_folder=config_folder)
        # 机械臂应用对象
		self.arm_app = ArmApplication(self.arm, config_folder=config_folder)
		# 创建关节状态发布者
		self.create_joint_state_publisher()
		# 创建视觉信息发布者
		self.create_marker_publisher()
		# 创建TF空间变换的发布者
		self.create_tf_publisher()
		# 发布工作台的TF静态变换
		self.publish_ws_board_tf()
		# 创建定时器
		self.timer = self.create_timer(0.05, self.timer_callback)
		# 绘制工作台
		self.draw_ws_board()

		# 创建获取机械臂状态服务
		self.create_get_arm_state_server()
		# 创建关节角度查询服务
		self.create_get_joint_angle_server()
		# 创建末端位姿查询服务
		self.create_get_tool_pose_server()
		# 创建设置机械臂阻尼模式的服务
		self.create_set_damping_server()
		# 创建夹爪控制的动作
		self.create_gripper_control_server()
		# 创建机械臂关节控制的动作
		self.create_joint_angle_control_server()
		# 创建机械臂位姿的动作
		self.create_set_tool_pose_server()
		# 创建控制末端移动到工作台坐标的动作
		self.create_move_to_workspace_server()
		# 创建机械臂物块抓取的动作
		self.create_grab_object_server()
		self._logger.info("五自由度机械臂服务器初始化完成")
	
	def get_ws_pose(self):
		# 获取工作台1的位姿
		T_arm2ws = self.arm_app.T_arm2ws
		ws_pose = Pose()
		ws_pose.set_transform_matrix(T_arm2ws, unit="mm")
		return ws_pose

	def draw_ws_board(self):
		'''绘制工作台'''
		ws_pose = self.get_ws_pose()
		# 填写Header
		now_msg = self.get_clock().now().to_msg()
		self.marker_msg.header.stamp = now_msg
		self.marker_msg.header.frame_id = "base_link"
		# 填写ID
		self.marker_msg.ns = "grab_cubic" # 填写命名空间
		self.marker_msg.id = self.MARKER_WS_01 # 填写Marker ID
		# 设置Marker类型为Mesh
		self.marker_msg.type = Marker.MESH_RESOURCE
		# 填写位姿
		x, y, z = ws_pose.get_position(unit="m")
		self.marker_msg.pose.position.x = float(x)
		self.marker_msg.pose.position.y = float(y)
		self.marker_msg.pose.position.z = float(z)
		q = ws_pose.get_quaternion()
		qx, qy, qz, qw = q.xyzw()
		self.marker_msg.pose.orientation.x = float(qx)
		self.marker_msg.pose.orientation.y = float(qy)
		self.marker_msg.pose.orientation.z = float(qz)
		self.marker_msg.pose.orientation.w = float(qw)
		# 填写尺度
		# 注: 必须是浮点数
		self.marker_msg.scale.x = 1.0
		self.marker_msg.scale.y = 1.0
		self.marker_msg.scale.z = 1.0
		# 填写颜色
		# 注: 必须是浮点数
		self.marker_msg.color.a = 1.0 
		self.marker_msg.color.r = 0.5
		self.marker_msg.color.g = 0.8
		self.marker_msg.color.b = 0.5
		# 填写MESH文件的路径
		self.marker_msg.mesh_resource = "package://arm5dof_description/meshes/ws_board.STL"
		# 发布Marker
		self.marker_publisher.publish(self.marker_msg)
    
	
	def create_joint_state_publisher(self):
		'''创建关节状态发布者'''
		# 创建关节状态信息
		self.joint_state_msg = JointState()
		# 创建发布者
		qos_profile = QoSProfile(depth=10)
		self.joint_state_publisher = self.create_publisher(JointState, \
			"joint_states", qos_profile)
		
	def create_marker_publisher(self):
		# 创建Marker的信息
		self.marker_msg = Marker()
		# 创建发布者
		self.marker_publisher = self.create_publisher(Marker, \
			"visualization_marker", 0)
	
	def create_tf_publisher(self):
		'''创建TF2发布者'''
		self.tf_static_publisher = tf2_ros.StaticTransformBroadcaster(self)
		self.tf_publisher = tf2_ros.TransformBroadcaster(self)
	
	def create_get_arm_state_server(self):
		'''创建机械臂状态信息服务'''
		self.get_arm_state_server = self.create_service(GetArmState, 
			"get_arm_state", self.get_arm_state_callback)
	
	def create_get_joint_angle_server(self):
		'''创建关节角度查询服务'''
		self.get_joint_angle_server = self.create_service(GetJointAngle,\
			"get_joint_angle", self.get_joint_angle_callback)
	
	def create_get_tool_pose_server(self):
		'''创建获取工具末端位姿服务'''
		self.get_tool_pose_server = self.create_service(GetToolPose,\
			"get_tool_pose", self.get_tool_pose_callback)

	def create_set_damping_server(self):
		'''创建设置机械臂为阻尼模式的服务'''
		self.set_damping_server = self.create_service(SetDamping,\
			"set_damping", self.set_damping_callback)

	def create_gripper_control_server(self):
		'''创建夹爪控制的动作服务器'''
		self.gripper_control_server = ActionServer(\
			self, GripperControl, "gripper_control", \
			self.gripper_control_callback)

	def create_joint_angle_control_server(self):
		'''创建机械臂关节角度控制服务器'''
		self.joint_angle_control_server = ActionServer(\
			self, JointAngleControl, "joint_angle_control", \
			self.joint_angle_control_callback)

	def create_set_tool_pose_server(self):
		'''创建机械臂末端位姿控制服务器'''
		self.set_tool_pose_server = ActionServer(\
			self, SetToolPose, "set_tool_pose", \
			self.set_tool_pose_callback)

	
	def create_move_to_workspace_server(self):
		'''创建控制末端移动到工作台坐标的动作'''
		self.move_to_workspace_server = ActionServer(\
			self, MoveToWorkspace, "move_to_workspace", \
			self.move_to_workspace_callback)

	def create_grab_object_server(self):
		'''创建物块抓取服务'''
		self.grab_object_server =  ActionServer(\
			self, GrabObject, "grab_object", \
			self.grab_object_callback)

	def publish_joint_state(self, is_query=True):
		'''发布关节状态信息'''
		# 填写Header
		now = self.get_clock().now()
		self.joint_state_msg.header.stamp = now.to_msg()
		# 填写关节名称
		self.joint_state_msg.name = self.arm.joint_name_list
		# 填写关节角度
		if is_query:
			angle_list = self.arm.get_joint_angle_list()
		else:
			# 使用轨迹中间点来作为查询到的关节角度
			# 因为频繁的查询角度 会导致轨迹执行卡顿
			angle_list = self.arm.get_target_joint_angle_list()
		
		self.joint_state_msg.position = [float(value) for value in angle_list]
		# 发布信息
		self.joint_state_publisher.publish(self.joint_state_msg)
	

	def publish_ws_board_tf(self):
		'''发布工作台的空间变换'''
		# 获取工作台1在世界坐标系下的位姿
		ws_pose = self.get_ws_pose()
		# 构造TF信息
		tf_msg = tf2_ros.TransformStamped()
		tf_msg.header.stamp = self.get_clock().now().to_msg()
		tf_msg.header.frame_id = "base_link"
		tf_msg.child_frame_id = "workspace"
		x, y, z = ws_pose.get_position()
		tf_msg.transform.translation.x = float(x)
		tf_msg.transform.translation.y = float(y)
		tf_msg.transform.translation.z = float(z)
		q = ws_pose.get_quaternion()
		qx, qy, qz, qw = q.xyzw()
		tf_msg.transform.rotation.x = float(qx)
		tf_msg.transform.rotation.y = float(qy)
		tf_msg.transform.rotation.z = float(qz)
		tf_msg.transform.rotation.w = float(qw)
		# 发布TF信息
		self.tf_static_publisher.sendTransform(tf_msg)

	def timer_callback(self, is_query=True):
		'''定时器回调函数'''
		# 发布关节状态信息
		self.publish_joint_state(is_query=is_query)


	def get_arm_state_callback(self, request, response):
		'''获取机械臂状态回调函数'''
		# 填写机械臂状态信息
		# - 因为是仿真，所以就伪造下串口信息
		response.state.is_uart_connected = self.arm.uart.isOpen()
		response.state.port_name = self.arm.device
		response.state.is_busy = not self.arm.is_stop()
		return response

	def get_joint_angle_callback(self, request, response):
		'''获取机械臂关节角度的回调函数'''
		# 创建关节角度列表的Message
		joint_angle_msg = response.joint_angle
		for joint_name in request.joint_name:
			# 查询关节角度
			joint_angle = self.arm.get_joint_angle(joint_name)
			# 填写响应信息
			joint_angle_msg.joint_name.append(joint_name)
			joint_angle_msg.joint_angle.append(joint_angle)
		return response
	
	def get_tool_pose_callback(self, request, response):
		'''获取工具位姿的回调函数'''
		# 获取工具坐标系位姿
		x, y, z, pitch, roll = self.arm.get_tool_pose()
		# 填写消息
		tool_pose_msg = ToolPose()
		tool_pose_msg.position.x = float(x)
		tool_pose_msg.position.y = float(y)
		tool_pose_msg.position.z = float(z)
		tool_pose_msg.pitch = float(pitch)
		tool_pose_msg.roll = float(roll)
		response.tool_pose = tool_pose_msg
		return response
	
	def set_damping_callback(self, request, response):
		'''设置阻尼模式回调函数'''
		# 设置阻尼模式的功率
		power = request.power
		# 设置机械臂为阻尼模式
		self.arm.set_damping(power)
		# 填写回传信息
		response.is_done = True
		return response
	
	def set_joint_angle_soft(self, *args):
		'''设置单个关节角度带 Minimum Jerk轨迹规划'''
		# 提取输入参数
		joint_name, joint_angle, max_power, T, callback = args
		self._logger.info(f"控制关节{joint_name} 运动到{joint_angle} 周期{T} 最大功 {max_power}")
		t_arr, theta_arr = self.arm.trajectory_plan(joint_name, joint_angle, T)
		# 按照轨迹去执行
		i = 0
		tn = len(t_arr)
		
		while rclpy.ok() and i < tn:
			t_start = time.time()
			next_theta = theta_arr[i]
			# 设置关节弧度
			self.arm.set_joint_angle(joint_name, next_theta, interval=0, max_power=max_power)
			t_end = time.time()
			m = int(math.ceil((t_end - t_start) / TRAJECTORY_DELTA_T))
			if m == 0:
				m = 1
			i += m
			# 补齐所需延迟等待的时间
			time.sleep(m*TRAJECTORY_DELTA_T - (t_end - t_start))
			# 回调函数执行
			if callback is not None and i%self.trajectory_callback_period == 0:
				callback()
		return theta_arr 
	
	def set_joint_angle_list_soft(self, *args):
		'''设置关节弧度2-带MinimumJerk 轨迹规划版本,需要阻塞等待.
		'''
		# 提取输入参数
		joint_name_list, joint_angle_list, T, callback = args
		# 获取关节个数
		joint_num = len(joint_name_list)
		# 构造目标关节角度字典
		thetas_dict = {}
		for i in range(joint_num):
			joint_name = joint_name_list[i]
			joint_angle = joint_angle_list[i]
			thetas_dict[joint_name] = joint_angle
		
		# theta序列
		theta_seq_dict = {}
		t_arr = None # 时间序列
		# 生成轨迹序列
		for joint_name, theta_e in thetas_dict.items():
			t_arr, theta_arr = self.arm.trajectory_plan(joint_name, theta_e, T)
			theta_seq_dict[joint_name] = theta_arr

		# 按照轨迹去执行
		i = 0
		tn = len(t_arr)
		
		while rclpy.ok() and i < tn:
			t_start = time.time()
			next_thetas = []
			for joint_name in joint_name_list:
				next_thetas.append(theta_seq_dict[joint_name][i])
			# 设置关节弧度
			self.arm.set_joint_angle_list(next_thetas, joint_name_list=joint_name_list,\
				interval=0, is_wait=False)
			t_end = time.time()
			m = int(math.ceil((t_end - t_start) / TRAJECTORY_DELTA_T))
			if m == 0:
				m = 1
			i += m
			# 补齐所需延迟等待的时间
			time.sleep(m*TRAJECTORY_DELTA_T - (t_end - t_start))
			# 回调函数执行
			if callback is not None and i%self.trajectory_callback_period==0:
				callback()
			
		return theta_seq_dict

	def set_gripper_angle(self, angle=0.0, max_power=3000, T=1.0, callback=None):
		'''设置夹爪角度'''
		thread_set_joint_angle = threading.Thread(target=self.set_joint_angle_soft,\
			args=("left_gripper_joint", angle, max_power, T, callback), daemon=True)
		thread_set_joint_angle.start()
		thread_set_joint_angle.join()

	def gripper_open(self, angle=np.pi/4, max_power=3000,  T=1.0, callback=None):
		'''夹爪打开'''
		self.set_gripper_angle(angle=angle, max_power=max_power, T=T, callback=callback)
	
	def gripper_close(self, angle=0.0, max_power=3000, T=1.0, callback=None):
		'''夹爪闭合'''
		self.set_gripper_angle(angle=angle, max_power=max_power, T=T, callback=callback)

	def gripper_control_callback(self, goal_handle):
		'''夹爪控制回调函数'''
		# 获取控制指令
		# - 目标角度
		target_angle = goal_handle.request.angle
		# - 控制周期
		period = goal_handle.request.period
		# - 最大输出功率
		max_power = goal_handle.request.max_power
		self._logger.info(f"开启夹爪控制 控制夹爪运动到: {target_angle}")

		# 创建反馈信息
		feedback_msg = GripperControl.Feedback()

		def callback():
			'''回调函数'''
			# 填写反馈信息
			feedback_msg.angle = self.arm.get_joint_angle("left_gripper_joint")
			# 发布反馈信息
			goal_handle.publish_feedback(feedback_msg)
			self._logger.info(f"发送反馈信息-夹爪角度: {feedback_msg.angle }")
			# 在执行Action的时候，定时器也不会执行
			# 所以手动设置下
			self.timer_callback(is_query=False)
			
		# 设置夹爪角度
		self.set_gripper_angle(target_angle, max_power=max_power, T=period, callback=callback)

		# 完成运动
		goal_handle.succeed()
		
		# 返回最终结果
		result_msg = GripperControl.Result()
		result_msg.angle = self.arm.get_joint_angle("left_gripper_joint")
		self._logger.info(f"夹爪动作完成-最终到达角度: {result_msg.angle}")
		return result_msg

	def joint_angle_control_callback(self, goal_handle):
		'''关节角度控制回调函数'''
		# 获取请求参数
		# - 关节名称列表
		joint_name_list = goal_handle.request.joint_name
		# - 关节角度列表
		joint_angle_list = goal_handle.request.joint_angle
		# - 默认是开启了Minimum Jerk轨迹规划
		# - 周期
		period = goal_handle.request.period
		# 打印日志
		self._logger.info(f"开启关节角度控制: ")
		self._logger.info(f" - 关节名称列表: {joint_name_list}")
		self._logger.info(f" - 关节角度列表: {joint_angle_list}")
		self._logger.info(f" - 控制周期: {period}")

		# 创建反馈信息
		feedback_msg = JointAngleControl.Feedback()
		feedback_msg.joint_name = joint_name_list
		def callback():
			'''回调函数'''
			# 填写反馈信息
			cur_joint_angle_list =  self.arm.get_joint_angle_list(joint_name_list=joint_name_list)
			feedback_msg.joint_angle = [float(v) for v in cur_joint_angle_list]
			# 发布反馈信息
			goal_handle.publish_feedback(feedback_msg)

			self._logger.info(f"发送关节角度信息反馈: ")
			self._logger.info(f"- 关节名称: {feedback_msg.joint_name}")
			self._logger.info(f"- 关节角度: {feedback_msg.joint_angle}")
			# 在执行Action的时候，定时器也不会执行
			# 所以手动设置下
			self.timer_callback(is_query=False)

		# 创建线程
		thread_set_joint_angle_list = threading.Thread(target=self.set_joint_angle_list_soft,\
			args=(joint_name_list, joint_angle_list, period, callback), daemon=True)
		thread_set_joint_angle_list.start()
		thread_set_joint_angle_list.join()

		# 完成运动
		goal_handle.succeed()
		
		# 返回最终结果
		result_msg = JointAngleControl.Result()
		result_msg.joint_name = joint_name_list
		result_joint_angle_list = self.arm.get_joint_angle_list(joint_name_list=joint_name_list)
		result_msg.joint_angle = [float(v) for v in result_joint_angle_list]
		self._logger.info(f"完成关节角度控制: ")
		self._logger.info(f"- 关节名称: {result_msg.joint_name}")
		self._logger.info(f"- 关节角度: {result_msg.joint_angle}")
		return result_msg
	

	def set_tool_pose(self, tool_posi=None, tool_pitch=None, \
		tool_roll=np.pi, pose_name=None, is_soft_move=True, T=1.0, \
		is_debug=False, callback=None):
		'''设置工具位姿'''
		# 根据名称填充位姿
		if pose_name is not None:
			# 使用位姿名称
			if pose_name in ARM_POSE_DICT.keys():
				x, y, z, tool_pitch, tool_roll = ARM_POSE_DICT[pose_name]
				tool_posi = [x, y, z]
			else:
				# 未知的位姿名称
				return False
		
		# 自动生成Pitch
		if tool_pitch is None:
			tool_pitch = self.arm.auto_gen_pitch(tool_posi)
		# 逆向运动学
		ret, thetas = self.arm.inverse_kinematic(\
			tool_posi, tool_pitch, tool_roll, is_debug=is_debug)
		# 判断是否有解
		if not ret:
			self._logger.warn('机械臂末端不能到达: {} Pitch={} Rool={}'.format(\
				tool_posi, tool_pitch, tool_roll))
			return False
		# 控制关节
		joint_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
		joint_angle_list = thetas
		# 创建线程
		thread_set_joint_angle_list = threading.Thread(target=self.set_joint_angle_list_soft,\
			args=(joint_name_list, joint_angle_list, T, callback), daemon=True)
		thread_set_joint_angle_list.start()
		thread_set_joint_angle_list.join()


	def set_tool_pose_callback(self, goal_handle):
		'''工具位姿控制回调函数'''
		# 获取输入参数
		# - 是否使用位姿名称
		use_pose_name = goal_handle.request.use_pose_name
		# - 位姿名称
		pose_name = goal_handle.request.pose_name
		# - 工具位姿
		tool_pose_msg = goal_handle.request.pose
		# - 自动生成俯仰角
		auto_gen_pitch = goal_handle.request.auto_gen_pitch
		# - 周期
		period = goal_handle.request.period
		
		# 反馈信息
		feedback_msg = SetToolPose.Feedback()
		joint_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
		# 定义回调函数
		def callback():
			'''回调函数'''
			# 填写反馈信息
			cur_joint_angle_list =  self.arm.get_joint_angle_list(joint_name_list=joint_name_list)
			feedback_msg.joint_angle.joint1 = float(cur_joint_angle_list[0])
			feedback_msg.joint_angle.joint2 = float(cur_joint_angle_list[1])
			feedback_msg.joint_angle.joint3 = float(cur_joint_angle_list[2])
			feedback_msg.joint_angle.joint4 = float(cur_joint_angle_list[3])
			feedback_msg.joint_angle.joint5 = float(cur_joint_angle_list[4])
			# 发布反馈信息
			goal_handle.publish_feedback(feedback_msg)

			self._logger.info(f"发送关节角度信息反馈: ")
			self._logger.info(f"- 关节名称: {str(feedback_msg.joint_angle)}")
			# 在执行Action的时候，定时器也不会执行
			# 所以手动设置下
			self.timer_callback(is_query=False)
		# 重新整理
		if use_pose_name:
			self._logger.info(f"运动到位姿名称: {pose_name}")
			self.set_tool_pose(pose_name=pose_name, T=period, callback=callback)
		else:
			# 获取目标值
			tool_posi = [tool_pose_msg.position.x, tool_pose_msg.position.y, tool_pose_msg.position.z]
			tool_pitch = None
			if not auto_gen_pitch:
				tool_pitch = tool_pose_msg.pitch
			tool_roll = tool_pose_msg.roll
			self._logger.info(f"运动到位姿:")
			self._logger.info(f"位置: {tool_posi} 俯仰角：{tool_pitch} 横滚角：{tool_roll}")
			self.set_tool_pose(tool_posi=tool_posi, tool_pitch=tool_pitch,\
				tool_roll=tool_roll, T=period, callback=callback)
		# 完成运动
		goal_handle.succeed()
		
		# 返回最终结果
		result_msg = SetToolPose.Result()
		result_msg.is_reach_goal = True
		x, y, z, pitch, roll = self.arm.get_tool_pose()
		result_msg.tool_pose.position.x = float(x)
		result_msg.tool_pose.position.y = float(y)
		result_msg.tool_pose.position.z = float(z)
		result_msg.tool_pose.pitch = float(pitch)
		result_msg.tool_pose.roll = float(roll)

		self._logger.info(f"完成工具位姿控制: ")
		self._logger.info(f"- 位置: {[x, y, z]}")
		self._logger.info(f"- 俯仰角: {pitch} 横滚角: {roll}")
		return result_msg

	def move_to_workspace(self, ws_posi, T=1.0, callback=None):
		'''运动到工作台上的坐标点'''
		# 将这个点转换为机械臂基坐标系下
		wx, wy, wz = ws_posi
		tool_posi = self.arm_app.tf_ws2arm(wx, wy, wz)
		self._logger.info(f"转换到机械臂基坐标系下:\n {tool_posi}")
		self.set_tool_pose(tool_posi=tool_posi, T=T, callback=callback)

	def move_to_workspace_callback(self, goal_handle):
		'''运动到工作台回调函数'''
		# 获取输入参数
		# - 获取工作台上的坐标
		wx = goal_handle.request.position.x
		wy = goal_handle.request.position.y
		wz = goal_handle.request.position.z
		ws_posi = [wx, wy, wz]
		# - 获取运动周期
		period = goal_handle.request.period
		self._logger.info(f"控制机械臂运动到工作台 {ws_posi}")
		self._logger.info(f"周期: {period}")

		# 反馈信息
		feedback_msg = MoveToWorkspace.Feedback()
		joint_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
		# 定义回调函数
		def callback():
			'''回调函数'''
			# 填写反馈信息
			cur_joint_angle_list = self.arm.get_joint_angle_list(joint_name_list=joint_name_list)
			feedback_msg.joint_angle.joint1 = float(cur_joint_angle_list[0])
			feedback_msg.joint_angle.joint2 = float(cur_joint_angle_list[1])
			feedback_msg.joint_angle.joint3 = float(cur_joint_angle_list[2])
			feedback_msg.joint_angle.joint4 = float(cur_joint_angle_list[3])
			feedback_msg.joint_angle.joint5 = float(cur_joint_angle_list[4])
			# 发布反馈信息
			goal_handle.publish_feedback(feedback_msg)
			self._logger.info(f"发送关节角度信息反馈: ")
			self._logger.info(f"- 关节名称: {str(feedback_msg.joint_angle)}")
			# 在执行Action的时候，定时器也不会执行
			# 所以手动设置下
			self.timer_callback(is_query=False)
		
		# 运动到工作台
		self.move_to_workspace(ws_posi, T=period, callback=callback)

		# 完成运动
		goal_handle.succeed()
		# 返回最终结果
		result_msg = MoveToWorkspace.Result()
		result_msg.is_reach_goal = True
		# 查询当前工具末端在工作台上的坐标
		tool_posi = self.arm.get_tool_pose()[:3]
		wx, wy, wz = self.arm_app.tf_arm2ws(*tool_posi)
		result_msg.position.x = wx
		result_msg.position.y = wy
		result_msg.position.z = wz
		return result_msg
	
	def grab_object(self, source_posi, target_posi, \
			gripper_open_angle, gripper_max_power, \
			z_lift_up, callback=None):
		'''抓取平台上的一个物块'''
		
		self.grab_stage_name = "1.夹爪张开"
		self.set_gripper_angle(angle=gripper_open_angle, T=1.0, callback=callback)
				
		self.grab_stage_name = "2. 运动到起始抓取点上方"
		wx, wy, wz = source_posi
		self.move_to_workspace([wx, wy, wz+z_lift_up], T=0.5, callback=callback)

		self.grab_stage_name = "3. 运动到起始抓取点"
		self.move_to_workspace([wx, wy, wz], T=0.3, callback=callback)
		
		self.grab_stage_name = "4. 夹爪闭合"
		self.gripper_close(angle=0.0, max_power=gripper_max_power, \
			T=1.0, callback=callback)

		self.grab_stage_name = "5. 抬起物块"
		self.move_to_workspace([wx, wy, wz+z_lift_up], T=0.4, callback=callback)

		self.grab_stage_name = "6. 运动到目标点上方" 
		wx, wy, wz = target_posi
		self.move_to_workspace([wx, wy, wz+z_lift_up], T=0.5, callback=callback)
		
		self.grab_stage_name = "7. 运动到目标点"
		self.move_to_workspace([wx, wy, wz], T=0.4, callback=callback)
		
		self.grab_stage_name = "8. 释放物块"
		self.set_gripper_angle(angle=gripper_open_angle, T=1.0, callback=callback)
		
		self.grab_stage_name = "9. 抬起夹爪，完成物块搬运" 
		wx, wy, wz = target_posi
		self.move_to_workspace([wx, wy, wz+z_lift_up], T=0.4, callback=callback)
	
	def grab_object_callback(self, goal_handle):
		'''抓取物块回调函数'''
		# 提取输入参数
		sx = goal_handle.request.source_position.x
		sy = goal_handle.request.source_position.y
		sz = goal_handle.request.source_position.z
		source_posi = [sx, sy, sz]
		tx = goal_handle.request.target_position.x
		ty = goal_handle.request.target_position.y
		tz = goal_handle.request.target_position.z
		target_posi = [tx, ty, tz]
		gripper_open_angle = goal_handle.request.gripper_open_angle
		gripper_max_power = goal_handle.request.gripper_max_power
		z_lift_up = goal_handle.request.z_lift_up

		self._logger.info("抓取物块")
		self._logger.info(f"- 起始位置: {source_posi}")
		self._logger.info(f"- 目标位置: {target_posi}")
		self._logger.info(f"- 夹爪张开角度: {gripper_open_angle}")
		self._logger.info(f"- 夹爪舵机最大输出功率: {gripper_max_power}")
		self._logger.info(f"- Z轴抬起高度: {z_lift_up}")
		# 反馈信息
		feedback_msg = GrabObject.Feedback()
		# 初始化阶段信息
		self.grab_stage_name = "开始进行物块抓取"
		# 回调函数
		def callback():
			'''回调函数'''
			# 填写反馈信息
			feedback_msg.stage_name = self.grab_stage_name
			# 发布反馈信息
			goal_handle.publish_feedback(feedback_msg)
			self._logger.info(f"物块抓取阶段: {feedback_msg.stage_name}")
			# 在执行Action的时候，定时器也不会执行
			# 所以手动设置下
			self.timer_callback(is_query=False)
		# 动作执行
		self.grab_object(source_posi, target_posi, \
			gripper_open_angle, gripper_max_power, \
			z_lift_up, callback=callback)

		# 完成运动
		goal_handle.succeed()
		# 返回最终结果
		result_msg = GrabObject.Result()
		result_msg.is_done = True
		return result_msg

def main(args=None):
	# 初始化rclpy
	print("hello")
	rclpy.init(args=args)
	# 创建服务器
	arm5dof_server = Arm5DoFUServoServer()
	# 开启多线程
	#thread = threading.Thread(target=rclpy.spin, args=(arm5dof_server, ), daemon=True)
	#thread.start()
	#thread.join()
	# 设置机械臂卸力
	arm5dof_server.arm.set_damping(1000)
	print("hello")
	# 关闭rclpy
	rclpy.shutdown()
main()
	
