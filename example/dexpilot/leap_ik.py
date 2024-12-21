#!/usr/bin/env python3
import time
import pybullet as p
import numpy as np
import rospy
import os

from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState

from dex_retargeting.retargeting_config import RetargetingConfig

"""
This takes the glove data, and runs inverse kinematics and then publishes onto LEAP Hand.

Note how the fingertip positions are matching, but the joint angles between the two hands are not.  :) 

Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
"""


class LeapPybulletIK:
    def __init__(self):
        rospy.init_node("leap_pyb_ik")
        # start pybullet
        # clid = p.connect(p.SHARED_MEMORY)
        # clid = p.connect(p.DIRECT)
        p.connect(p.GUI)
        # diable gravity
        # load right leap hand
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.is_left = rospy.get_param("~isLeft", False)
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [3, 4, 8, 9, 13, 14, 18, 19]
        self.short_idx = [3, 4, 8, 9, 13, 14, 18, 19, 23, 24]
        if self.is_left:
            path_src = os.path.join(path_src, "leap_hand_mesh_left/robot_pybullet.urdf")
            ##You may have to set this path for your setup on ROS2
            self.LeapId = p.loadURDF(
                path_src,
                [-0.05, -0.03, -0.25],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
            )

            self.pub_hand = rospy.Publisher(
                "/leaphand_node/cmd_allegro_left", JointState, queue_size=10
            )
            self.sub_skeleton = rospy.Subscriber("/manus_left", PoseArray, self.get_glove_data)
        else:
            # path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
            path_src = os.path.join(path_src, "../../assets/leap_hand_dottip_right_wo_col_retarget.urdf")
            ##You may have to set this path for your setup on ROS2
            self.LeapId = p.loadURDF(
                path_src,
                # [-0.05, -0.03, -0.125],
                # [0.125, 0.05, 0],
                # [0.030, 0, 0.020], 
                [0,0,0],
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
            )

            self.pub_hand = rospy.Publisher(
                "/leaphand_node/cmd_allegro_right", JointState, queue_size=10
            )
            self.sub_skeleton = rospy.Subscriber("/manus_right", PoseArray, self.get_glove_data)

        self.numJoints = p.getNumJoints(self.LeapId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()
        self.create_retargeting_solver()

        # get active Joint names from pybullet
        pybullet_joint_names = []
        self.pybullet_joint_ids = []
        for i in range(self.numJoints):
            joint_info = p.getJointInfo(self.LeapId, i)
            if joint_info[2] != p.JOINT_FIXED:
                pybullet_joint_names.append(joint_info[1].decode("utf-8"))
                self.pybullet_joint_ids.append(joint_info[0])

        print(f"pybullet dof joint ids: {self.pybullet_joint_ids}")
        print(f"pybullet dof joint names: {pybullet_joint_names}")
        retargeting_joint_names = self.retargeting.joint_names
        print(f"retargeting joint names: {retargeting_joint_names}")
        self.retargeting_to_pybullet = np.array([retargeting_joint_names.index(name) for name in pybullet_joint_names]).astype(int)


    def create_retargeting_solver(self):
        RetargetingConfig.set_default_urdf_dir("../../assets")
        retarget_cfg = RetargetingConfig.load_from_file("./cfg/leap_hand_right.yml")
        self.retargeting = retarget_cfg.build()


    def create_target_vis(self):
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]

        self.ballMbt = []
        for i in range(0, 4):
            self.ballMbt.append(
                p.createMultiBody(
                    baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition
                )
            )  # for base and finger tip joints
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(
                self.ballMbt[i], -1, no_collision_group, no_collision_mask
            )
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1])
        p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1])
        p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])
        p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])

    def update_target_vis(self, hand_pos, hand_quat):
        # p.resetBasePositionAndOrientation(self.LeapId, [0, 0, 0], hand_quat[0])

        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[0])
        p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos[3], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[1])
        p.resetBasePositionAndOrientation(self.ballMbt[1], hand_pos[5], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[2])
        p.resetBasePositionAndOrientation(self.ballMbt[2], hand_pos[7], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[3])
        p.resetBasePositionAndOrientation(self.ballMbt[3], hand_pos[1], current_orientation)

    def get_glove_data(self, pose):
        # gets the data converts it and then computes IK and visualizes
        poses = pose.poses
        # print(f"pose array length: {len(poses)}")
        hand_pos_vis = []
        hand_pos = []
        hand_quat = []
        # for i in range(0, 10):
        for pose in poses:
            hand_pos_vis.append(
                # [
                #     poses[self.short_idx[i]].position.x * self.glove_to_leap_mapping_scale * 1.15,
                #     poses[self.short_idx[i]].position.y * self.glove_to_leap_mapping_scale,
                #     -poses[self.short_idx[i]].position.z * self.glove_to_leap_mapping_scale,
                # ]
                [
                    pose.position.z * 0.95 * self.glove_to_leap_mapping_scale, 
                    -pose.position.x * 1.2 * self.glove_to_leap_mapping_scale,
                    pose.position.y * self.glove_to_leap_mapping_scale,
                ]
            )
            hand_pos.append(
                [
                    pose.position.z * 0.95, 
                    -pose.position.x * 1.2,
                    pose.position.y,
                ]
            )
            hand_quat.append(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )

        # hand_pos[2][0] = hand_pos[2][0] - 0.02  this isn't great because they won't oppose properly
        # hand_pos[3][0] = hand_pos[3][0] - 0.02
        # hand_pos[6][0] = hand_pos[6][0] + 0.02
        # hand_pos[7][0] = hand_pos[7][0] + 0.02
        # hand_pos[2][1] = hand_pos[2][1] + 0.002

        # hand_pos[4][1] = hand_pos[4][1] + 0.002
        # hand_pos[6][1] = hand_pos[6][1] + 0.002
        # self.compute_IK_pybullet(hand_pos)
        self.compute_retargeting(np.array(hand_pos))
        self.update_target_vis(np.array(hand_pos_vis)[self.short_idx], hand_quat)

    
    def compute_retargeting(self, hand_pos: np.ndarray):
        p.stepSimulation()
        # retargeting
        retargeting_type = self.retargeting.optimizer.retargeting_type
        indices = self.retargeting.optimizer.target_link_human_indices

        ref_hand_pos = hand_pos[indices, :]

        tic = time.perf_counter()
        qpos = self.retargeting.retarget(ref_hand_pos)
        tac = time.perf_counter()

        rospy.loginfo(f"Retargeting time: {tac - tic}")

        jointPoses = qpos[self.retargeting_to_pybullet]
        # print(f"jointPoses: {jointPoses}")
        rospy.loginfo(f"jointPoses: {jointPoses}")

        # combined_jointPoses = (
        #     jointPoses[0:4]
        #     + (0.0,)
        #     + jointPoses[4:8]
        #     + (0.0,)
        #     + jointPoses[8:12]
        #     + (0.0,)
        #     + jointPoses[12:16]
        #     + (0.0,)
        # )
        # combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        # for i in range(20):
        # for i, joint_index in enumerate(self.pybullet_joint_ids):
        #     p.setJointMotorControl2(
        #         bodyIndex=self.LeapId,
        #         jointIndex=joint_index,
        #         controlMode=p.POSITION_CONTROL,
        #         targetPosition=jointPoses[i],
        #         # targetVelocity=0,
        #         # force=500,
        #         # positionGain=0.2,
        #         # velocityGain=1.6,
        #     )
        
        # reset joint positions
        for i, joint_index in enumerate(self.pybullet_joint_ids):
            p.resetJointState(self.LeapId, joint_index, targetValue=jointPoses[i])


        # map results to real robot TODO: index mapping
        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        # real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_robot_hand_q[0:4] = jointPoses[0:4]
        real_robot_hand_q[4:8] = jointPoses[4:8]
        real_robot_hand_q[8:12] = jointPoses[8:12]
        real_robot_hand_q[12:16] = jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        stater = JointState()
        stater.position = [float(i) for i in real_robot_hand_q]
        self.pub_hand.publish(stater)

        # total_time = 0
        # for i, joint_pos in enumerate(hand_pos):
        #     if retargeting_type == "POSITION":
        #         indices = indices
        #         ref_value = joint_pos[indices, :]
        #     else:
        #         origin_indices = indices[0, :]
        #         task_indices = indices[1, :]
        #         ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        #     tic = time.perf_counter()
        #     qpos = self.retargeting.retarget(ref_value)
        #     tac = time.perf_counter()
        #     total_time += tac - tic
        # return total
    
    
    
    def compute_IK_pybullet(self, hand_pos):
        p.stepSimulation()

        rightHandIndex_middle_pos = hand_pos[2]
        rightHandIndex_pos = hand_pos[3]

        rightHandMiddle_middle_pos = hand_pos[4]
        rightHandMiddle_pos = hand_pos[5]

        rightHandRing_middle_pos = hand_pos[6]
        rightHandRing_pos = hand_pos[7]

        rightHandThumb_middle_pos = hand_pos[0]
        rightHandThumb_pos = hand_pos[1]

        leapEndEffectorPos = [
            rightHandIndex_middle_pos,
            rightHandIndex_pos,
            rightHandMiddle_middle_pos,
            rightHandMiddle_pos,
            rightHandRing_middle_pos,
            rightHandRing_pos,
            rightHandThumb_middle_pos,
            rightHandThumb_pos,
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )

        combined_jointPoses = (
            jointPoses[0:4]
            + (0.0,)
            + jointPoses[4:8]
            + (0.0,)
            + jointPoses[8:12]
            + (0.0,)
            + jointPoses[12:16]
            + (0.0,)
        )
        combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        for i in range(20):
            p.setJointMotorControl2(
                bodyIndex=self.LeapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        # map results to real robot
        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        # real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_robot_hand_q[0:4] = jointPoses[0:4]
        real_robot_hand_q[4:8] = jointPoses[4:8]
        real_robot_hand_q[8:12] = jointPoses[8:12]
        real_robot_hand_q[12:16] = jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        stater = JointState()
        stater.position = [float(i) for i in real_robot_hand_q]
        self.pub_hand.publish(stater)


def main(args=None):
    # rclpy.init(args=args)
    try:
        node = LeapPybulletIK()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
