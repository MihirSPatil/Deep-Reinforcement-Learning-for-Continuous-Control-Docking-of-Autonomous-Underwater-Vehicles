import rospy
import numpy as np
from numpy import cos, sin
from scipy.spatial.transform import Rotation as R


def quat2euler_angle(pose_data):
    roll, pitch, yaw = R.from_quat([pose_data.orientation.x,
                                    pose_data.orientation.y,
                                    pose_data.orientation.z,
                                    pose_data.orientation.w]).as_euler('xyz')

    return roll, pitch, yaw


def coordinate_frame_transform(roll, pitch, yaw, frame="world2body"):
    """
    Returns the rotation matrix for transforming between the world and body coordinates or vice-versa,
    depending on the value of 'frame'.
    Takes the roll, pitch and yaw as the inputs
    """

    rot_mat = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    angular_rot_mat = np.array(([1, 0, -sin(pitch)],
                                [0, cos(roll), cos(pitch) * sin(roll)],
                                [0, -sin(roll), cos(roll) * cos(pitch)]))

    if frame.lower() == "world2body":
        return rot_mat.T, angular_rot_mat

    if frame.lower() == "body2world":
        return rot_mat, np.linalg.pinv(angular_rot_mat)


def modelstate2numpy(data, mode='pose'):
    if mode.lower() == 'pose':
        auv_pose = data.pose[-1]
        roll, pitch, yaw = quat2euler_angle(auv_pose)
        auv_pose = np.round(np.array([auv_pose.position.x,
                                      auv_pose.position.y,
                                      auv_pose.position.z,
                                      roll,
                                      pitch,
                                      yaw]), 2)
        return auv_pose

    if mode.lower() == 'vel':
        auv_vel = data.twist[-1]
        auv_vel = np.array([auv_vel.linear.x,
                            auv_vel.linear.y,
                            auv_vel.linear.z,
                            auv_vel.angular.x,
                            auv_vel.angular.y,
                            auv_vel.angular.z])
        return auv_vel
