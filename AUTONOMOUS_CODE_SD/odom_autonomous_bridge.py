"""
Odometry bridge utilities for autonomous runtime.

Purpose:
- Keep odometry math and camera-to-odom transforms in one reusable place.
- Provide a lightweight goal-seeking controller that can be enabled in staged merges.
- Preserve current autonomous behavior unless caller explicitly uses bridge outputs.

Coordinate frames:
- Camera point: (x_c, y_c, z_c) from RealSense deprojection, in meters.
- Robot/base pose: (x, y, theta) in odom frame.
- Odom point: transformed world point in odom frame, in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin


@dataclass
class OdomPose:
    """Robot pose estimate in odom frame."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class OdomAutonomousBridge:
    """
    Shared odometry helper used by the autonomous host runtime.

    The bridge is intentionally independent from serial/device code.
    Callers feed measured linear/angular velocity and can request local goal control.
    """

    def __init__(
        self,
        cam_offset_x_m: float = -0.64,
        cam_offset_y_m: float = 0.0,
        cam_offset_z_m: float = 0.2,
        kp_v: float = 0.5,
        kp_w: float = 0.5,
        max_v: float = 0.30,
        max_w: float = 0.60,
        distance_tolerance_m: float = 0.05,
        forward_is_negative: bool = True,
    ) -> None:
        # Camera-to-base translation from validated field calibration.
        self.cam_offset_x_m = cam_offset_x_m
        self.cam_offset_y_m = cam_offset_y_m
        self.cam_offset_z_m = cam_offset_z_m

        # Local controller settings for odom-goal motion.
        self.kp_v = kp_v
        self.kp_w = kp_w
        self.max_v = max_v
        self.max_w = max_w
        self.distance_tolerance_m = distance_tolerance_m
        self.forward_is_negative = forward_is_negative

        self.pose = OdomPose()
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_active = False

        self.targ_lin_vel = 0.0
        self.targ_ang_vel = 0.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def set_goal(self, goal_x: float, goal_y: float) -> None:
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_active = True

    def clear_goal(self) -> None:
        self.goal_active = False
        self.targ_lin_vel = 0.0
        self.targ_ang_vel = 0.0

    def update_pose_from_feedback(self, meas_lin_vel: float, fuse_ang_vel: float, dt_s: float) -> None:
        """Integrate wheel/IMU fused motion feedback into odom pose."""
        if dt_s <= 0.0:
            return
        self.pose.x += meas_lin_vel * cos(self.pose.theta) * dt_s
        self.pose.y += meas_lin_vel * sin(self.pose.theta) * dt_s
        self.pose.theta += fuse_ang_vel * dt_s
        # Keep heading wrapped to [-pi, pi].
        self.pose.theta = atan2(sin(self.pose.theta), cos(self.pose.theta))

    def compute_goal_velocity(self) -> tuple[float, float]:
        """
        Compute velocity toward active goal in odom frame.

        Returns:
        - (lin_vel, ang_vel) in the same sign convention as configured.
        """
        if not self.goal_active:
            self.targ_lin_vel = 0.0
            self.targ_ang_vel = 0.0
            return self.targ_lin_vel, self.targ_ang_vel

        dx = self.goal_x - self.pose.x
        dy = self.goal_y - self.pose.y
        distance_error = hypot(dx, dy)

        if distance_error < self.distance_tolerance_m:
            self.clear_goal()
            return self.targ_lin_vel, self.targ_ang_vel

        target_heading = atan2(dy, dx)
        heading_error = atan2(
            sin(target_heading - self.pose.theta),
            cos(target_heading - self.pose.theta),
        )

        cmd_w = self.kp_w * heading_error
        direction_alignment = max(0.0, cos(heading_error))
        cmd_v_mag = self.kp_v * distance_error * direction_alignment

        cmd_w = self._clamp(cmd_w, -self.max_w, self.max_w)
        cmd_v_mag = self._clamp(cmd_v_mag, 0.0, self.max_v)

        lin_sign = -1.0 if self.forward_is_negative else 1.0
        self.targ_lin_vel = lin_sign * cmd_v_mag
        self.targ_ang_vel = cmd_w
        return self.targ_lin_vel, self.targ_ang_vel

    def transform_cam_to_odom(self, coords_cam: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        Transform a camera-frame point to odom frame using current pose.

        The fixed camera->base rotation follows your validated odometry scripts.
        """
        x_c, y_c, z_c = coords_cam

        # camera_link -> base_link
        # R_c_b = [[0,0,1],[-1,0,0],[0,-1,0]]
        x_b = z_c + self.cam_offset_x_m
        y_b = -x_c + self.cam_offset_y_m
        z_b = -y_c + self.cam_offset_z_m

        # base_link -> odom using current robot heading and position.
        c = cos(self.pose.theta)
        s = sin(self.pose.theta)
        x_o = c * x_b - s * y_b + self.pose.x
        y_o = s * x_b + c * y_b + self.pose.y
        z_o = z_b
        return x_o, y_o, z_o
