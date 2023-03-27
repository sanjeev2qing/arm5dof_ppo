'''
自由度机械臂Python SDK
----------------------------------------------
@作者: 朱林
@公司: 湖南创乐博智能科技有限公司
@邮箱: zhulin@loborobot.com
@官方网站: https://www.loborobot.com/
'''
import os
import time 		# 时间
import logging 		# 日志管理
import serial 		# 串口通信
import numpy as np 	# 科学计算 矩阵计算

# 自定义库
from config import * # 导入配置文件
from uservo import UartServoManager # 导入串口舵机管理器
from arm5dof_kinematic import Arm5DoFKinematic # 运动学
from minimum_jerk import minimum_jerk_plan, minimum_jerk_seq # 轨迹规划算法

class Arm5DoFUServo:
	'''五自由度机械臂 | 真机SDK'''
	# 旋转关节名称列表
	joint_name_list = ["joint1", "joint2", "joint3", "joint4", "joint5", "left_gripper_joint"]
	# 等待超时阈值
	wait_timeout = 10
	# 轨迹执行回调函数执行周期, 几个时间块执行一次
	trajectory_callback_period = 20
	
	def __init__(self, device:str=DEVICE_PORT_DEFAULT, is_init_pose:bool=True, \
     		config_folder="config", is_debug=False):
		'''机械臂初始化'''
		# 创建串口对象
		try:
			self.is_debug = is_debug
			self.device = device
			self.uart = serial.Serial(port=device, baudrate=115200,\
					 parity=serial.PARITY_NONE, stopbits=1,\
					 bytesize=8,timeout=0)
			# 创建串口舵机管理器
			self.uservo = UartServoManager(self.uart, srv_num=SERVO_NUM)
			# 创建运动学求解器
			self.kinematic = Arm5DoFKinematic()
			self.theta_lowerb = self.kinematic.JOINT_ANGLE_LOWERB.tolist()
			self.theta_upperb = self.kinematic.JOINT_ANGLE_UPPERB.tolist()
			# 添加夹爪的配置
			self.theta_lowerb.append(0.0)
			self.theta_upperb.append(np.pi/2)
   			# Z轴误差补偿系数
			file_path = os.path.join(config_folder, "z_error_polyfit.txt")
			self.z_error_polyfit = np.loadtxt(file_path,  delimiter=",")
   			# 关节目标值
			self.target_joint_angle_dict = {}
			# 机械臂位姿初始化
			if is_init_pose:
				self.home()
			
		except serial.SerialException as e:
			logging.error('该设备不存在,请重新填写UART舵机转接板的端口号')
			# 退出
			# exit(-1)
			quit()
	def joint_name2servo_id(self, joint_name):
		'''将关节名称转换为舵机ID'''
		return JOINT_SERVO_ID_DICT[joint_name]
	
	def home(self):
		'''回归机械零点'''
		self.set_tool_pose(pose_name="home", T=2.0, is_wait=True, is_soft_move=True)

	def set_servo_velocity(self, speed:float):
		'''设置舵机的转速 单位°/s'''
		self.uservo.mean_dps = max(min(abs(speed), SERVO_SPEED_MAX), SERVO_SPEED_MIN)
	
	def set_servo_angle(self, angles, is_wait:bool=False, callback=None):
		'''设置舵机的原始角度'''
		if type(angles) == list:
			for srv_idx, angle in enumerate(angles):
				logging.info('设置舵机角度, 舵机#{} 目标角度 {:.1f}'.format(srv_idx, angle))
				self.uservo.set_servo_angle(int(srv_idx), float(angle))
		elif type(angles) == dict:
			for srv_idx, angle in angles.items():
				logging.info('设置舵机角度, 舵机#{} 目标角度 {:.1f}'.format(srv_idx, angle))
				self.uservo.set_servo_angle(int(srv_idx), float(angle))
		if wait:
			# 等待舵机角度到达目标角度
			self.wait(callback=callback) 

	def set_joint_angle(self, joint_name, theta, interval=None, \
			max_power=0, is_wait=False, callback=None):
		'''设置关节角度'''
		# 更新关节目标值
		self.target_joint_angle_dict[joint_name] = theta
		# 舵机ID
		servo_id = self.joint_name2servo_id(joint_name)
		# 检查关节角度是否合法
		# 检查弧度的范围约束
		theta = min(max(self.theta_lowerb[servo_id], theta),\
			self.theta_upperb[servo_id])
		# 关节角度转换为驱动器舵机原始角度
		# 根据关节的弧度计算出舵机的原始角度
		angle = JOINT2SERVO_K[servo_id]*theta + JOINT2SERVO_B[servo_id] 
		self.uservo.set_servo_angle(int(servo_id), float(angle), interval=interval, power=max_power)
		# 等待
		if is_wait:
			return self.wait(callback=callback)
		else:
			return True
	
	def set_joint_angle_list(self, theta_list, joint_name_list=None, \
			interval=None, max_power=0, is_wait=False, callback=None):
		'''批量设置关节角度'''
		# 关节个数
		joint_num = len(theta_list)
		for i in range(joint_num):
			# 获取关节名称
			joint_name = None
			if joint_name_list is None:
				joint_name = self.joint_name_list[i]
			else:
				joint_name = joint_name_list[i]
			# 设置关节角度
			theta = theta_list[i]
			self.set_joint_angle(joint_name, theta, interval=interval,\
				max_power=max_power, is_wait=False)
		# 等待
		if is_wait:
			return self.wait(callback=callback)
		else:
			return True
	
	def trajectory_plan(self, joint_name, theta_e:float, T:float, w_s:float=0.0, w_e:float=0.0, a_s:float=0.0, a_e:float=0):
		'''Minimum Jerk轨迹规划'''
		# 获取当前关节的
		theta_s = self.get_joint_angle(joint_name)
		# print("起始弧度 {} 中止弧度 {}".format(theta_s, theta_e))
		c = minimum_jerk_plan(theta_s, theta_e, w_s, w_e, a_s, a_e, T)
		t_arr, theta_arr = minimum_jerk_seq(T, c, delta_t=TRAJECTORY_DELTA_T)
		return t_arr, theta_arr
	
	def set_joint_angle_soft(self, joint_name, joint_angle, T=1.0, max_power=0, callback=None):
		'''设置单个关节角度带 Minimum Jerk轨迹规划'''
		t_arr, theta_arr = self.trajectory_plan(joint_name, joint_angle, T)
		# 按照轨迹去执行
		i = 0
		tn = len(t_arr)
		
		while True:
			t_start = time.time()
			if i >= tn:
				break
			next_theta = theta_arr[i]
			# 设置关节弧度
			self.set_joint_angle(joint_name, next_theta, interval=0, max_power=max_power)
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
	
	def set_joint_angle_list_soft(self, joint_angle_list, joint_name_list=None, T:float=1.0, callback=None):
		'''设置关节弧度2-带MinimumJerk 轨迹规划版本,需要阻塞等待.
		'''
		joint_num = len(joint_angle_list)
		if joint_name_list is None:
			joint_name_list = self.joint_name_list[:joint_num]
		# 将thetas转换为dict类型
		theta_dict = {}
		for i in range(joint_num):
			theta_dict[joint_name_list[i]] = joint_angle_list[i]
		
		# theta序列
		theta_seq_dict = {}
		t_arr = None # 时间序列
		# 生成轨迹序列
		for joint_name, theta_e in theta_dict.items():
			t_arr, theta_arr = self.trajectory_plan(joint_name, theta_e, T=T)
			theta_seq_dict[joint_name] = theta_arr

		# 按照轨迹去执行
		i = 0
		tn = len(t_arr)
		
		while True:
			t_start = time.time()
			if i >= tn:
				break
			next_thetas = []
			for joint_name in joint_name_list:
				next_thetas.append(theta_seq_dict[joint_name][i])
			# 设置关节弧度
			self.set_joint_angle_list(next_thetas, \
				joint_name_list=joint_name_list, interval=0, is_wait=False)
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
		
		# 等待运动结束
		return theta_seq_dict

	def get_servo_angle(self, joint_name):
		'''读取舵机原始角度'''
		servo_id = JOINT_SERVO_ID_DICT[joint_name]
		servo_angle = self.uservo.query_servo_angle(servo_id)
		return servo_angle
	
	def get_servo_angle_list(self, joint_name_list=None):
		'''获取原始舵机角度列表'''
		if joint_name_list is None:
			return [self.uservo.query_servo_angle(srv_idx) \
				for srv_idx in range(SERVO_NUM)]
		else:
			servo_raw_angle_list = []
			for joint_name in joint_name_list:
				servo_angle = self.get_joint_angle(joint_name)
				servo_raw_angle_list.append(servo_angle)
			return servo_raw_angle_list
	
	def get_joint_angle(self, joint_name):
		'''获取关节角度'''
		# 舵机ID 
		servo_id = JOINT_SERVO_ID_DICT[joint_name]
		# 查询舵机原始角度
		servo_angle = self.get_servo_angle(joint_name)
		# 将舵机角度转换为关节弧度
		joint_angle = (servo_angle - JOINT2SERVO_B[servo_id]) / JOINT2SERVO_K[servo_id]
		# 获取关节状态
		return joint_angle
	
	def get_joint_angle_list(self, joint_name_list=None):
		'''批量获取关节角度'''
		theta_list = [] # 关节弧度列表
		if joint_name_list is not None:
			for joint_name in joint_name_list:
				theta_list.append(self.get_joint_angle(joint_name))
		else:
			for joint_name in self.joint_name_list:
				theta = self.get_joint_angle(joint_name)
				# 添加到关节列表里面
				theta_list.append(theta)
		return np.float64(theta_list)

	def get_target_joint_angle_list(self):
		'''获取关节角度目标值'''
		theta_list = []
		for joint_name in self.joint_name_list:
			if not joint_name in self.target_joint_angle_dict:
				self.target_joint_angle_dict[joint_name] = self.get_joint_angle(joint_name)
			theta_list.append(self.target_joint_angle_dict[joint_name])
		return theta_list
	
	def get_tool_pose(self, return_type="XYZ_Pitch_Roll"):
		'''获取工具坐标系的位姿'''
		# 获取当前的关节角度列表
		cur_joint_angle = self.get_joint_angle_list()[:5]
		# 返回的数据格式: [x6, y6, z6, pitch, roll]
		return self.kinematic.forward_kinematic_v2(cur_joint_angle,\
      		return_type=return_type)
	
	def inverse_kinematic(self, tool_posi, tool_pitch=np.pi/2, tool_roll=np.pi, is_debug=False):
		'''逆向运动学
		@tool_posi: 工具末端的位置
		@tool_pitch: 工具末端的俯仰角
		'''
		# 获取上次的关节角度
		last_joint_angle = self.get_joint_angle_list()[:5]
		if is_debug:
			print(f"上一次的关节角度: {last_joint_angle}")
		# 获取候选关节角度列表
		tx, ty, tz = tool_posi
		candi_joint_angle_list = self.kinematic.inverse_kinematic(\
	  		tx, ty, tz, tool_pitch, tool_roll, last_joint_angle=last_joint_angle, is_debug=is_debug)
		# 选取一个
		if len(candi_joint_angle_list) == 0:
			return False, None
		else:
			# TODO 选取一个最合适的解
			return True, candi_joint_angle_list[-1]
	
	def adjust_tool_posi(self, p_tool:list):
		'''调整工具坐标'''
		x, y, z = p_tool
		r = np.sqrt(x**2 + y**2)
		k, b = self.z_error_polyfit
		z_error = k*r + b
		return [x, y, z+z_error]
	
	def auto_gen_pitch(self, p_tool:list):
		'''自动生成俯仰角'''
		x, y, z = p_tool
		x0 = np.sqrt(x**2 + y**2)
		return PITCH_PANEL_A*x0 + PITCH_PANEL_B*z + PITCH_PANEL_C

	def set_tool_pose(self, tool_posi=None, tool_pitch=None, \
		tool_roll=np.pi, pose_name=None, is_soft_move=True, T=1.0, \
		is_wait=False, is_debug=False, callback=None):
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
		# 自动修正目标位置
		tool_posi = self.adjust_tool_posi(tool_posi)
		# 自动生成Pitch
		if tool_pitch is None:
			tool_pitch = self.auto_gen_pitch(tool_posi)
		# 逆向运动学
		ret, thetas = self.inverse_kinematic(\
			tool_posi, tool_pitch, tool_roll, is_debug=is_debug)
		# 判断是否有解
		if not ret:
			logging.warn('机械臂末端不能到达: {} 俯仰角={} 横滚角={}'.format(\
				tool_posi, tool_pitch, tool_roll))
			return False
		# 控制关节
		# print(f"设置关节角度:  {thetas}")
		if is_soft_move:
			self.set_joint_angle_list_soft(thetas, T=T, callback=callback)
		else:
			return self.set_joint_angle_list(thetas, is_wait=is_wait, callback=callback)
	
	def set_gripper_angle(self, angle=0.0, max_power=3000, \
			is_soft_move=True, T=1.0, is_wait=True, callback=None):
		'''设置夹爪角度'''
		if is_soft_move:
			self.set_joint_angle_soft("left_gripper_joint", angle, \
				max_power=max_power, T=T, callback=callback)
		else:
			self.set_joint_angle("left_gripper_joint", angle, \
				max_power=max_power, is_wait=is_wait, callback=callback)
		
	def gripper_open(self, angle=np.pi/4, max_power=3000, is_soft_move=True, \
			T=1.0, is_wait=True, callback=None):
		'''夹爪张开'''
		self.set_gripper_angle(angle=angle, max_power=max_power, \
			is_soft_move=is_soft_move, T=T, is_wait=is_wait, callback=callback)
		
	def gripper_close(self, angle=0, max_power=3000, is_soft_move=True,\
			T=1.0, is_wait=True, callback=None):
		'''夹爪闭合'''
		self.set_gripper_angle(angle=angle, max_power=max_power, \
			is_soft_move=is_soft_move, T=T, is_wait=is_wait, callback=callback)
	
	def is_stop(self):
		'''返回当前机械臂末端是否静止'''
		return self.uservo.is_stop()
	
	def wait(self, callback=None):
		'''机械臂等待'''
		t_start = time.time()
		while not self.uservo.is_stop():
			t_cur = time.time() 
			if t_cur - t_start > self.wait_timeout:
				# 超时
				return False
			# 执行回调函数
			if callback is not None:
				callback()
			# 等待10ms
			time.sleep(0.01)
		return True
	
	def set_damping(self, power=0):
		'''设置为阻尼模式'''
		for servo_id in range(SERVO_NUM):
			self.uservo.set_damping(servo_id, power)