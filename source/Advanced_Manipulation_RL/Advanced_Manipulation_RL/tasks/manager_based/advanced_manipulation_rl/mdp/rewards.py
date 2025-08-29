# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:

    """Reward the robot for reaching the drawer handle using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the handle.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)

    # get current x and y direction of the handle
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]

    # make sure gripper aligns with the handle
    # # in this case, the z direction of the gripper should be close to the -x direction of the handle
    # # and the x direction of the gripper should be close to the -y direction of the handle
    # # dot product of z and x should be large
    # align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


    # get current x, y, z direction of the handle
    handle_x, handle_y, handle_z = handle_mat[..., 0], handle_mat[..., 1], handle_mat[..., 2]
    # get current x, y, z direction of the gripper
    ee_tcp_x, ee_tcp_y, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 1], ee_tcp_rot_mat[..., 2]

    # Strongly align the z direction of the gripper with the z direction of the handle (outward from door)
    align_z_axis = torch.bmm(ee_tcp_z.unsqueeze(1), handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # Keep the original alignments as well
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    # Increase the weight for the z-axis alignment (outward direction)
    return 2.0 * (torch.sign(align_z_axis) * align_z_axis**2) + 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)

def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # # Check if hand is in a graspable pose
    # is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    # Z-axis: left finger above, right finger below
    z_check = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])
    # X-axis: knob between fingers
    x_min = torch.minimum(lfinger_pos[:, 0], rfinger_pos[:, 0])
    x_max = torch.maximum(lfinger_pos[:, 0], rfinger_pos[:, 0])
    x_check = (handle_pos[:, 0] > x_min) & (handle_pos[:, 0] < x_max)
    # Y-axis: knob between fingers
    y_min = torch.minimum(lfinger_pos[:, 1], rfinger_pos[:, 1])
    y_max = torch.maximum(lfinger_pos[:, 1], rfinger_pos[:, 1])
    y_check = (handle_pos[:, 1] > y_min) & (handle_pos[:, 1] < y_max)

    # All axes must be satisfied for a true grasp
    is_graspable = z_check & x_check & y_check

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.015) -> torch.Tensor: # offset was 0.04
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))


def grasp_handle(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the drawer given by the joint position of the drawer.

    The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()

    return (is_graspable + 1.0) * drawer_pos


def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Multi-stage bonus for opening the drawer.

    Depending on the drawer's position, the reward is given in three stages: easy, medium, and hard.
    This helps the agent to learn to open the drawer in a controlled manner.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()

    open_easy = (drawer_pos > 0.05) * 0.5
    open_medium = (drawer_pos > 0.5) * is_graspable
    open_hard = (drawer_pos > 1.5) * is_graspable

    return open_easy + open_medium + open_hard


# def ee_behind_knob(env: "ManagerBasedRLEnv") -> torch.Tensor:
#     """Reward for keeping the end effector (TCP) behind the knob (negative z of knob frame)."""
#     handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
#     handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]
#     ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
#     handle_rot = matrix_from_quat(handle_quat)
#     knob_z = handle_rot[..., 2]
#     knob_to_ee = ee_tcp_pos - handle_pos
#     proj = torch.bmm(knob_to_ee.unsqueeze(1), knob_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     return (proj < 0).float()

def ee_at_knob_pose(env: "ManagerBasedRLEnv", z_offset: float = -0.07, tol: float = 0.015) -> torch.Tensor:
    """Reward for keeping the end effector (TCP) at a specific -z offset and x/y=0 in the knob frame."""
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_rot = matrix_from_quat(handle_quat)  # (..., 3, 3)
    # Transform ee_tcp_pos to knob frame
    knob_to_ee = ee_tcp_pos - handle_pos  # (..., 3)
    knob_to_ee_knobframe = torch.matmul(handle_rot.transpose(-1, -2), knob_to_ee.unsqueeze(-1)).squeeze(-1)
    # Want x=0, y=0, z=z_offset (z negative)
    x_ok = torch.abs(knob_to_ee_knobframe[..., 0]) < tol
    y_ok = torch.abs(knob_to_ee_knobframe[..., 1]) < tol
    z_ok = torch.abs(knob_to_ee_knobframe[..., 2] - z_offset) < tol
    return (x_ok & y_ok & z_ok).float()



