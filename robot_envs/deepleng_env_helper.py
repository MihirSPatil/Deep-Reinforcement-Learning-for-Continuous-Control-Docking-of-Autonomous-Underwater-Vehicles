import numpy as np
from numpy import cos, sin, tan
from tf.transformations import euler_from_quaternion, quaternion_from_euler, euler_matrix, quaternion_matrix


# Methods that the TrainingEnvironment will need.

def rotation_from_quat(x,y,z,w):
    """
    return rotation matrix from quaternion
    """
    return quaternion_matrix([x, y, z, w])[:3, :3]

def angular_transform(roll, pitch, yaw, frame="world2body"):
    """
    Returns the rotation matrix for transforming between the world and body coordinates or vice-versa,
    depending on the value of 'frame'.
    Takes the roll, pitch and yaw as the inputs
    Note that body2world has a singularity at pitch = pi/2
    """

    angular_rot_mat = np.array(([1, 0, -sin(pitch)],
                                [0, cos(roll), cos(pitch) * sin(roll)],
                                [0, -sin(roll), cos(roll) * cos(pitch)]))

    if frame.lower() == "world2body":
        return np.array(([1, 0, -sin(pitch)],
                         [0, cos(roll), cos(pitch) * sin(roll)],
                         [0, -sin(roll), cos(roll) * cos(pitch)]))

    if frame.lower() == "body2world":
        return np.array([[1., sin(roll)*tan(pitch), cos(roll)*tan(pitch)],
                         [0.,            cos(roll),           -sin(roll)],
                         [0., sin(roll)/cos(pitch), cos(roll)/cos(pitch)]])

def modelstate2numpy(data, mode='pose'):

    if mode.lower() == 'pose':
        auv_pose = data.pose[-1]
        roll, pitch, yaw = euler_from_quaternion([auv_pose.orientation.x,
                                                  auv_pose.orientation.y,
                                                  auv_pose.orientation.z,
                                                  auv_pose.orientation.w])
        auv_pose = np.array([auv_pose.position.x,
                             auv_pose.position.y,
                             auv_pose.position.z,
                             roll,
                             pitch,
                             yaw])
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

def get_nose_position(position, orientation, nose_in_body):
    """
     Gets the position of the AUV nose in world frame
     
     Parameters:
     position: np.array, [x,y,z] position of auv center in world coordinates
     orientation: np.array, [x,y,z,w] orientation of auv as quaternion
     nose_in_body: np.array, position of the auv nose in body coordinates
    """
    rot = rotation_from_quat(orientation[0],
                             orientation[1],
                             orientation[2],
                             orientation[3])
    return position + rot.dot(nose_in_body)

def linear_to_body(world_lin_vel, orientation):
    """
    Transform linear velocity from world to body coordinates

    Parameters:
     world_lin_vel: np.array, [x',y',z'] linear velocity in world coordinates
     orientation: np.array, [x,y,z,w] orientation of auv as quaternion
    """
    return euler_from_quaternion(orientation).T.dot(world_lin_vel)

def linear_to_world(body_lin_vel, orientation):
    """
    Transform linear velocity from body to world coordinates
    Parameters:
     body_lin_vel: np.array, [u,v,w] linear velocity in body coordinates
     orientation: np.array, [x,y,z,w] orientation of auv as quaternion
    """
    return euler_from_quaternion(orientation).dot(body_lin_vel)


def angular_to_body(world_ang_vel, orientation):
    """
    Transform angular velocities from world to body coordinates
    Parameters:
     world_ang_vel: np.array, [roll',pitch',yaw'] angular velocity in world coordinates
     orientation: np.array, [x,y,z,w] orientation of auv as quaternion
    """
    roll, pitch, yaw = euler_from_quaternion(orientation)
    return angular_transform(roll, pitch, yaw, frame="world2body").dot(world_ang_vel)

def angular_to_world(body_ang_vel, orientation):
    """
    Transform angular velocities from body to world coordinates
    Parameters:
     body_and_vel: np.array, [p,q,r] angular velocity in body coordinates
     orientation: np.array, [x,y,z,w] orientation of auv as quaternion
    """
    roll, pitch, yaw = euler_from_quaternion(orientation)
    return angular_transform(roll, pitch, yaw, frame="body2world").dot(body_ang_vel)


if __name__ == "__main__":
    quat = [0.0, 0.0, 0.38268343, 0.92387953]
    roll, pitch, yaw = 0, 0, np.pi/4
    print(euler_from_quaternion(quat))
    print(angular_transform(roll, pitch, yaw))
    print(np.allclose(euler_from_quaternion(quat), [roll, pitch, yaw]))
    position = np.array([5,5,0])
    nose_in_body = np.array([1,0,0])
    print(get_nose_position(position, quat, nose_in_body))
