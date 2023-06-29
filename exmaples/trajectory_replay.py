#!/usr/bin/python


from Go1_Traj_Replay import Go1TrajReplay
import sys
import time
import math
import numpy as np


sys.path.append('../lib/python/amd64')
import robot_interface as sdk





def main():
    
    LOWLEVEL  = 0xff

    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    
    go1 = Go1TrajReplay(udp, safe, cmd, state, control_freq = 400)
    
    go1.stand()
    
    go1.sit()
    
    play_traj = np.load('trajs/play_controlled_60.npy')
    
    for i in range(play_traj.shape[0]):
        if go1.joint_angle_sanity_check(play_traj[i]) == False:
            print('Joint angle out of range!')
            break
    
    go1.trajectory_replay(play_traj, replay_frequency=60)
    
    go1.sit()
    
    
if __name__ == '__main__':
    main()
    