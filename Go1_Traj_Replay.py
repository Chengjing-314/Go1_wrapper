import sys
import time
import math
import numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk
from scipy.interpolate import CubicSpline


class Go1TrajReplay():
    
    NUM_JOINTS = 12
    NUM_LEGS = 4
    DEF_SLEEP_TIME = 0.002
    JOINT_LIMIT = np.array([         # Hip, Thigh, Calf
        [-1.047,    -0.663,      -2.9],  # MIN
        [1.047,     2.966,       -0.837]  # MAX
    ])
    STAND = np.array(([
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171,
        -0.02452479861676693, 0.8545529842376709, -1.675719976425171
    ]))

    SIT = np.array([
        -0.27805507, 1.1002517, -2.7185173,
        0.307049, 1.0857971, -2.7133338,
        -0.263221, 1.138222, -2.7211301,
        0.2618303, 1.1157601, -2.7110581
    ])

    D = {'FR_0': 0, 'FR_1': 1, 'FR_2': 2,
         'FL_0': 3, 'FL_1': 4, 'FL_2': 5,
         'RR_0': 6, 'RR_1': 7, 'RR_2': 8,
         'RL_0': 9, 'RL_1': 10, 'RL_2': 11}
    
    def __init__(self, udp, safe, cmd, state, control_freq,  max_stable_time=100, sanity_check=True):
        
            
        self.udp = udp
        self.safe = safe
        self.cmd = cmd
        self.state = state
        self.max_stable_time = max_stable_time
        self.control_freq = control_freq
        self.sanity_check = sanity_check
        self._current_pose = self._get_stable_pose()
     
        print("Initialization Done")
        
    
    def _get_stable_pose(self):
        """This function retrieve stable position of the legged robot.
           Please ensure that the robot's join angle is within the bound at the very beginning

        Returns:
            numpy.ndarray: a (12,1) numpy array of robot's current joint angle and position. 
        """
        prev_pose = self.get_current_pose()
        motion_time = 0
        while True:
            time.sleep(Go1TrajReplay.DEF_SLEEP_TIME)
            cur_pose = np.zeros(12, dtype=np.float32)
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            for i in range(self.NUM_JOINTS):
                cur_pose[i] = self.state.motorState[i].q
            self.udp.Send()
            
            if self.joint_angle_sanity_check(cur_pose) and np.all(np.abs(cur_pose - prev_pose) < 0.001):
                break
            
            prev_pose = cur_pose
            motion_time += 1

            if motion_time >= 10:
                self.safe.PowerProtect(self.cmd, self.state, 1)
        
        return cur_pose
    
    def stand(self, interpolation_duration=2, standing_duration=1, interpolation_method="linear"):
        """Make the robot stand. 

        Args:
            interpolation_duration (int, optional): Time to interpolate to position. Defaults to 2 second.
            standing_duration (int, optional): Time for maintaining the pose. Defaults to 1 second.
            interpolation_method (str, optional): Method for interpolation('cubic' or 'linear'). Defaults to "linear".
        """
        
        self.move_to_pose_def(self.STAND, interpolation_duration=interpolation_duration, extra_duration=standing_duration, interpolation_method=interpolation_method)
        
    
    def sit(self, interpolation_duration=2, sitting_duration=1, interpolation_method="linear"):
        """Make the robot sit.

        Args:
            interpolation_duration (int, optional): Time to move to target position. Defaults to 2 seconds.
            sitting_duration (int, optional): Time for maintaining the pose. Defaults to 1 second.
            interpolation_method (str, optional): Method for interpolation('cubic' or 'linear'). Defaults to "linear".
        """
        self.move_to_pose_def(self.SIT, interpolation_duration=interpolation_duration, extra_duration=sitting_duration, interpolation_method=interpolation_method)
        
    
    def move_to_pose_def(self, target_position, interpolation_duration=2, extra_duration=1,  interpolation_method="linear"):
        """Move the whole trajectory to intend poses

        Args:
            target_position (numpy.ndarray): a (12,) or (12,1) ndarray represent that target pose of the 12 joints.
            interpolation_duration (int, optional): Time to move to position. Defaults to 2 seconds.
            extra_duration (int, optional): Time to maintain the target position. Defaults to 1 second.
            interpolation_method (str, optional): Method for interpolation('cubic' or 'linear'). Defaults to "linear".
        """
        interpolation_steps = int(interpolation_duration * self.control_freq)
        extra_steps = int(extra_duration * self.control_freq)
        tota_steps = interpolation_steps + extra_steps
        
        self._current_pose = self._get_stable_pose()
        
        for motion_time in range(tota_steps):
            begin_time = time.time()
            
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            next_pose = self.interpolation_by_rate(self._current_pose, target_position, rate = motion_time / interpolation_steps, interpolation_method=interpolation_method)
            next_pose = next_pose.reshape((self.NUM_JOINTS, 1))
            for i in range(self.NUM_JOINTS):
                self.cmd.motorCmd[i].q = next_pose[i]
                self.cmd.motorCmd[i].dq = 0
                self.cmd.motorCmd[i].Kp = 60
                self.cmd.motorCmd[i].Kd = 3.0
                self.cmd.motorCmd[i].tau = 0 # no torque control

            self.udp.SetSend(self.cmd)
            self.udp.Send()
            self.safe.PowerProtect(self.cmd, self.state, 1)
            
            done_time = time.time()
            time_compensation = np.fmax(0, 1 / self.control_freq - (done_time - begin_time))
            time.sleep(time_compensation)
        
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        
    def move_to_interpolated_pose(self, target_position):
        """This method is used to make small movement toward the target_position. 
           Intend to give user more power and flexibility in changing the movement. 
           Shoule be paried with interpolation by rate.

        Args:
            target_position (numpy.ndarray): a (12,) or (12,1) array that represent the target
            of the joint position. 
        """
        
        begin_time = time.time()
        if self.sanity_check:
            assert self.joint_angle_sanity_check(target_position), "JOINT ANGLE OUT OF BOUNDS, STOPPING"
            assert target_position.shape == (self.NUM_JOINTS, ), "WRONG SHAPE OF TARGET POSITION, STOPPING"
        
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        
        for i in range(self.NUM_JOINTS):
            self.cmd.motorCmd[i].q = target_position[i]
            self.cmd.motorCmd[i].dq = 0
            self.cmd.motorCmd[i].Kp = 60
            self.cmd.motorCmd[i].Kd = 3.0
            self.cmd.motorCmd[i].tau =0
        
        self.safe.PowerProtect(self.cmd, self.state, 1)
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        done_time = time.time()
        time_compensation = np.fmax(0, 1 / self.control_freq - (done_time - begin_time))
        time.sleep(time_compensation)
                
    def get_current_pose(self):
        """get the current position of the robot

        Returns:
            numpy.ndarray: a (12,) numpy array represent the momentary pose.
        """
        cur_pose = np.zeros(12, dtype=np.float32)
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        for i in range(self.NUM_JOINTS):
            cur_pose[i] = self.state.motorState[i].q
        self.udp.Send()
        time.sleep(self.DEF_SLEEP_TIME)
        return cur_pose
       
        
    def joint_angle_sanity_check(self, joint_angles, num_legs=4):
        """Check sanity of the joint angle

        Args:
            joint_angles (numpy.ndarray): a (12,) or (12,1) numpy ndarray represent the angle to be checked.
            num_legs (int, optional): numer of legs to check. Defaults to 4.

        Returns:
            boolean: if the constraint are satisfied. 
        """
        
        assert joint_angles.shape == (3 * num_legs, ) or joint_angles.shape == (3 * num_legs, 1), f"joint angle dimension num_legs * 3 must match with num_legs {num_legs}"
        
        jag = joint_angles.reshape(num_legs,3)
        
        check_res = np.all(jag > self.JOINT_LIMIT[0]) and np.all(jag < self.JOINT_LIMIT[1])
        
        return check_res 
    
    
    def interpolation_complete_traj(self, start_pose, target_pose, num_legs = 4,  num_steps=100, interpolation_method="linear"):
        """Interpolate the whole trajectory from start pose to target pose. 

        Args:
            start_pose (numpy.ndarray): starting pose of the robot
            target_pose (numpy.ndarray): target pose of the robot
            num_legs (int, optional): number of legs to look at. Defaults to 4.
            num_steps (int, optional): number of steps between start pose and target pose. Defaults to 100.
            interpolation_method (str, optional):interpolation method to use('linear' or 'cubic'). Defaults to "linear".

        Raises:
            ValueError: If Interpolation method not exist

        Returns:
            numpy.ndarray: a 4 * numsteps * 3 array. 
        """
        if self.sanity_check:
            assert start_pose.shape == (3 * num_legs,) or start_pose.shape == (3 * num_legs,1), f"start_pose dimension num_legs * 3 must match with num_legs {num_legs}"
            assert target_pose.shape == (3 * num_legs,) or target_pose.shape == (3 * num_legs,1), f"target_pose dimension num_legs * 3 must match with num_legs {num_legs}"
            assert self.joint_angle_sanity_check(start_pose, num_legs), "START JOINT ANGLES OUT OF BOUNDS, STOPPING INTERPOLATION"
            assert self.joint_angle_sanity_check(target_pose, num_legs), "TARGET JOINT ANGLES OUT OF BOUNDS, STOPPING INTERPOLATION"

        start_pose = np.array(start_pose).reshape(num_legs,3)
        target_pose = np.array(target_pose).reshape(num_legs,3)
        
        interpolated_poses = np.zeros((num_legs, num_steps, 3))
        
        steps = np.linspace(0, 1, num_steps)
        
        if interpolation_method == "linear":
            for i in range(start_pose.shape[0]):
                interpolated_poses[i] = start_pose[i] + steps.reshape(num_steps,1) * (target_pose[i] - start_pose[i])
        
        elif interpolation_method == "cubic":
            for i in range(start_pose.shape[0]):
                cs = CubicSpline(np.array([0, 1]), np.array([start_pose[i], target_pose[i]]))
                interpolated_poses[i] = cs(steps)
        
        else :
            raise ValueError(f"Interpolation method {interpolation_method} not supported")
            
        
        return interpolated_poses
    

    def interpolation_by_rate(self, start_pose, target_pose, num_legs = 4, rate = 0.5, interpolation_method = "linear"):
        """Interpolation the pose by rate

        Args:
            start_pose (numpy.ndarray): starting pose of the robot
            target_pose (numpy.ndarray): target pose of the robot
            num_legs (int, optional): number of legs on the robot to interpolate. Defaults to 4.
            rate (float, optional): the rate to interpolation, number between 0 and 1. Defaults to 0.5.
            interpolation_method (str, optional): interpolation_method (str, optional):interpolation method to use('linear' or 'cubic'). Defaults to "linear".

        Raises:
            ValueError: If Interpolation method not exist

        Returns:
            numpy.ndarray: a 4 * numsteps * 3 array. 
        """
        if self.sanity_check:
            assert start_pose.shape == (3 * num_legs,) or start_pose.shape == (3 * num_legs,1), f"start_pose dimension num_legs * 3 must match with num_legs {num_legs}"
            assert target_pose.shape == (3 * num_legs,) or target_pose.shape == (3 * num_legs,1), f"target_pose dimension num_legs * 3 must match with num_legs {num_legs}"
            assert self.joint_angle_sanity_check(start_pose), "START JOINT ANGLES OUT OF BOUNDS, STOPPING INTERPOLATION"
            assert self.joint_angle_sanity_check(target_pose), "TARGET JOINT ANGLES OUT OF BOUNDS, STOPPING INTERPOLATION"

        rate = np.fmax(np.fmin(rate, 1), 0)
        
        start_pose = np.array(start_pose).reshape(num_legs,3)
        target_pose = np.array(target_pose).reshape(num_legs,3)
        
        interpolated_poses = np.zeros((num_legs, 3))
        
        if interpolation_method == "linear":
            for i in range(start_pose.shape[0]):
                interpolated_poses[i] = start_pose[i] * (1 - rate) + target_pose[i] * rate
        
        elif interpolation_method == "cubic":
            for i in range(start_pose.shape[0]):
                cs = CubicSpline(np.array([0, 1]), np.array([start_pose[i], target_pose[i]]))
                interpolated_poses[i] = cs(rate)
        
        else :
            raise ValueError(f"Interpolation method {interpolation_method} not supported")
        
        if num_legs == 1:
            interpolated_poses = interpolated_poses.reshape(3)
        
        return interpolated_poses
    
    
    def reshape_interpolated_poses(self, interpolated_poses, num_legs = 4):
        """reshape the returned interpolated poses to num_steps, num_legs * 3

        Args:
            interpolated_poses (numpy.ndarray): 

        Returns:
            numpy.ndarray: reshaped array
        """
    
        num_steps = interpolated_poses.shape[1]
        
        return np.transpose(interpolated_poses, [1,0,2]).reshape(num_steps, num_legs * 3)
    
    def trajectory_replay(self, trajectories, replay_frequency = 100, maintain_last_pose = True):
        assert trajectories.shape[1] == 12, "Must control all joints and shape n * 12"
        
        start_pose = trajectories[0]
        
        self.move_to_pose_def(start_pose, interpolation_duration=5, extra_duration=2)
        
        # maintain the last pose
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        
        for i in range(trajectories.shape[0]-1):
            start_time = time.time()
            self.move_to_interpolated_pose(trajectories[i+1])
            time_compensation = np.fmax(0, 1 / replay_frequency - (time.time() - start_time))
            time.sleep(time_compensation)
        
        # maintain the last pose
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        
        if not maintain_last_pose:
            self.sit()