import pytest
import numpy as np
from openai_ros.robot_envs.transform_utils import rotation_from_quat, linear_to_body, linear_to_world

@pytest.fixture
def orientation():
    return np.array([0.0, 0.26, 0.0, -0.97])

@pytest.fixture
def world_lin_vel():
    return np.array([0.481231439797, float(2.78177729957e-5), 0.266818616932])

@pytest.fixture
def body_lin_vel():
    return np.array([float(5.50166313e-1),  float(2.78177730e-5), float(-9.63851271e-3)])


def test_linear_to_world(world_lin_vel, body_lin_vel, orientation):
    assert np.allclose(rotation_from_quat(orientation[0],
                                          orientation[1],
                                          orientation[2],
                                          orientation[3]).dot(body_lin_vel), world_lin_vel)


def test_linear_to_body(world_lin_vel, body_lin_vel, orientation):
    assert np.allclose(rotation_from_quat(orientation[0],
                                          orientation[1],
                                          orientation[2],
                                          orientation[3]).T.dot(world_lin_vel), body_lin_vel)
