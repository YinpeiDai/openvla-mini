from copy import deepcopy
from dataclasses import dataclass
import os
from typing import List, Optional, Union
import cv2
import imageio
import numpy as np
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=4, suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"


# TASK = RLBENCH_TASKS
# TOLERANCE_DICT = {
#     "light_bulb_in": {"front": 10, "left_shoulder": 20, "right_shoulder": 20},
#     "place_cups": {"front": 10, "left_shoulder": 10, "right_shoulder": 10},
#     "put_groceries_in_cupboard": {"front": 20, "left_shoulder": 25, "right_shoulder": 25},
#     "put_money_in_safe": {"front": 5, "left_shoulder": 20, "right_shoulder": 20},
#     "stack_cups": {"front": 8, "left_shoulder": 8, "right_shoulder": 8},
# }
# DEFAULT_TOLERANCE = {"front": 5, "left_shoulder": 5, "right_shoulder": 8}


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray
    gripper_open: bool

    def __post_init__(self):
        self.orientation = self.orientation / np.linalg.norm(self.orientation)
        if isinstance(self.gripper_open, np.ndarray):
            qpos_1, qpos_2 = self.gripper_open
            if abs(qpos_1) > 0.035 and abs(qpos_2) > 0.035:
                self.gripper_open = True
            else:
                self.gripper_open = False

    def __repr__(self):
        return f"Pose(position={self.position}, orientation={self.orientation}, gripper_open={self.gripper_open})"

class ReticleBuilder:
    def __init__(self, scene_bound=None, resolution=256, use_waypoint=False, use_next_action=False):
        self.image_size = resolution
        self.scene_bound = scene_bound  # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.use_waypoint = use_waypoint
        self.use_next_action = use_next_action
        self.last_center, self.last_Zh = None, None

        ##### Colors Setting #####

        # fix camera
        self.fixcam_line_color, self.fixcam_line_thickness = (0, 255, 0), 1
        self.fixcam_line_color_penetrate, self.fixcam_line_thickness_penetrate = (255, 0, 255), 1
        self.fixcam_line_color_ee2target = (255, 255, 0)
        self.fixcam_circle_ee_color_open, self.fixcam_ee_color_close = (255, 0, 0), (0, 0, 255)

        # wrist camera
        self.wrscam_center_color_open, self.wrscam_center_color_close = (255, 0, 0), (0, 0, 255)
        self.wrscam_reticle_color = (0, 255, 0)
        self.wrscam_line_color_ee2target = (255, 255, 0)

    @staticmethod
    def _3d_to_2d(point_3d, extrinsic_matrix, intrinsic_matrix):
        """
        Project a 3D point to 2D image
        """
        # Step 1: Transform 3D point to camera coordinates
        R = extrinsic_matrix[:3, :3]  # Rotation matrix
        T = extrinsic_matrix[:3, 3]  # Translation vector

        # Convert target point and point cloud to camera coordinates
        point_camera = np.dot(R, point_3d) + T

        X_c, Y_c, Z_c = point_camera  # Target point in camera coordinates

        # Step 2: Project target point to image plane
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        u_target = int(fx * X_c / Z_c + cx)
        v_target = int(fy * Y_c / Z_c + cy)
        return u_target, v_target, Z_c

    @staticmethod
    def _check_visibility(u_target, v_target, Z_c, camera_depth, image_width, image_height, threshold=0):
        """
        Check if the point is visible in the camera.
        -1: behind the camera
        -2: out of image bounds
        -3: occluded by some object
        """

        if Z_c <= 0:
            return -1

        if not (0 <= u_target < image_width and 0 <= v_target < image_height):
            return -2

        region = camera_depth[v_target - 1 : v_target + 1, u_target - 1 : u_target + 1]
        if region.size == 0:
            return -2

        if region.min() < Z_c + threshold:
            return -3

        return 1

    def _find_stop_point(
        self,
        position,
        orientation,
        camera_extrinsics,
        camera_intrinsics,
        camera_depth,
        image_width,
        image_height,
        tolerance=0,
        step_width=0.01,
    ):
        """
        Find the stop point of the shooting line
        start from the position, and move along the orientation, until it's not visible or a pc point is hit

        tolerance: the number of invisible points that are allowed
        """
        # change quat to rotation axis
        r = R.from_quat(orientation)
        orientation = r.as_matrix() @ np.array([0, 0, 1])
        points, points_uv, visibility = [], [], []
        step_width = step_width
        next_p = deepcopy(position)

        # shoot a line, add a small step and check if the point is visible
        while True:
            next_p += step_width * orientation
            # if (
            #     next_p[0] > self.scene_bound[3]
            #     or next_p[0] < self.scene_bound[0]
            #     or next_p[1] > self.scene_bound[4]
            #     or next_p[1] < self.scene_bound[1]
            #     or next_p[2] > self.scene_bound[5]
            #     or next_p[2] < self.scene_bound[2]
            # ):
            #     # out of bound
            #     break

            u_target, v_target, Z_c = self._3d_to_2d(next_p, camera_extrinsics, camera_intrinsics)
            is_visible = self._check_visibility(u_target, v_target, Z_c, camera_depth, image_width, image_height)

            if tolerance == 0:
                if is_visible < 0:
                    break
                points.append(next_p)
                points_uv.append((u_target, v_target))
                visibility.append(True)
            else:
                if is_visible < 0:
                    if len(visibility) > 2 * tolerance:
                        break

                if sum([1 for v in visibility[: 2 * tolerance] if v < 0]) > tolerance:
                    break
                points.append(next_p)
                points_uv.append((u_target, v_target))
                visibility.append(is_visible)

        return points, points_uv, visibility

    @staticmethod
    def _find_span(li, min_length=3):
        # return the start and end index of the span of 1s
        if sum([1 for v in li if v == 1]) < min_length:
            return []

        spans = []
        current_start = None
        current_length = 0

        for i, num in enumerate(li):
            if num == 1:
                if current_start is None:
                    current_start = i
                current_length += 1
            else:
                if current_length >= min_length:
                    spans.append((current_start, i - 1))
                current_start = None
                current_length = 0

        # Check the last sequence in case it reaches the end
        if current_length >= min_length:
            spans.append((current_start, len(li) - 1))
        return spans


    @staticmethod
    def get_rotated_line(center, angle, line_length):
        # find out the start point
        start_length = line_length // 6
        start_uv = (int(center[0] + start_length * np.cos(angle)), int(center[1] + start_length * np.sin(angle)))
        end_uv = (int(center[0] + line_length * np.cos(angle)), int(center[1] + line_length * np.sin(angle)))
        return start_uv, end_uv


    def render_on_fix_camera(
        self,
        camera_rgb,
        camera_depth,
        camera_extrinsics,
        camera_intrinsics,
        target_pose: Pose, # target pose to be drawn, it can be the next pose or the current pose
        gripper_pose: Pose,  # current gripper pose
        tolerance=10,
    ) -> np.ndarray:

        u, v, z = self._3d_to_2d(target_pose.position, camera_extrinsics, camera_intrinsics)

        visibility = self._check_visibility(u, v, z, camera_depth, self.image_size, self.image_size)

        if self.use_waypoint:
            if visibility in [-1, -2]:
                return camera_rgb, visibility

            _, points_uv, vv = self._find_stop_point(
                position=target_pose.position,
                orientation=target_pose.orientation,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
                camera_depth=camera_depth,
                image_width=self.image_size,
                image_height=self.image_size,
            )

            if points_uv:
                uv = points_uv[-1]
                color = self.fixcam_line_color if target_pose.gripper_open else self.fixcam_line_color_penetrate
                thickness = (
                    self.fixcam_line_thickness if target_pose.gripper_open else self.fixcam_line_thickness_penetrate
                )
                # draw a shooting line
                camera_rgb = cv2.line(camera_rgb, (u, v), uv, color, thickness)

            # draw a line from current gripper pose to the next pose
            current_u, current_v, _ = self._3d_to_2d(gripper_pose.position, camera_extrinsics, camera_intrinsics)
            camera_rgb = cv2.line(camera_rgb, (current_u, current_v), (u, v), self.fixcam_line_color_ee2target, 1)

            # draw a small circle on the current gripper pose
            camera_rgb = cv2.circle(camera_rgb, (current_u, current_v), 3, self.fixcam_line_color_ee2target, -1)

            # plot a point on the image indicating the next pose
            pose_color = self.fixcam_circle_ee_color_open if target_pose.gripper_open else self.fixcam_ee_color_close
            camera_rgb = cv2.circle(camera_rgb, (u, v), 3, pose_color, -1)
            return camera_rgb, visibility

        else:  # continuous control
            if visibility in [-1, -2]:
                return camera_rgb, visibility

            _, points_uv, vis = self._find_stop_point(
                position=target_pose.position,
                orientation=target_pose.orientation,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
                camera_depth=camera_depth,
                image_width=self.image_size,
                image_height=self.image_size,
                tolerance=tolerance,
            )
            
            # print("fix vis:", vis)

            # draw a shooting line from gripper to the stopping point, when the gripper is open
            if points_uv and target_pose.gripper_open:
                spans = self._find_span(vis)
                for start_id, end_id in spans[:1]:
                    camera_rgb = cv2.line(
                        camera_rgb,
                        points_uv[start_id],
                        points_uv[end_id],
                        self.fixcam_line_color,
                        self.fixcam_line_thickness,
                    )

            # draw a shooting line that penetrates the object, when the gripper is closed
            if points_uv and not target_pose.gripper_open:
                spans = self._find_span(vis)
                for start_id, end_id in spans:
                    camera_rgb = cv2.line(
                        camera_rgb,
                        points_uv[start_id],
                        points_uv[end_id],
                        self.fixcam_line_color_penetrate,
                        self.fixcam_line_thickness_penetrate,
                    )

            # plot a point on the image indicating the next pose
            pose_color = self.fixcam_circle_ee_color_open if target_pose.gripper_open else self.fixcam_ee_color_close
            camera_rgb = cv2.circle(camera_rgb, (u, v), 3, pose_color, -1)
            return camera_rgb, visibility


    def render_on_wst_camera(
        self,
        wrist_camera_rgb,
        wrist_camera_depth,
        wrist_camera_extrinsics,
        wrist_camera_intrinsics,
        target_pose: Pose, # target pose to be drawn, it can be the next pose or the current pose
        gripper_pose: Pose, # current gripper pose
        tolerance=15,
    ) -> np.ndarray:

        u_target, v_target, Z_c = self._3d_to_2d(target_pose.position, wrist_camera_extrinsics, wrist_camera_intrinsics)

        image_width = self.image_size
        image_height = self.image_size

        visibility = self._check_visibility(u_target, v_target, Z_c, wrist_camera_depth, image_width, image_height)

        if visibility in [-1, -2]:
            return wrist_camera_rgb, visibility
        
        _, points_uv, vis = self._find_stop_point(
            position=target_pose.position,
            orientation=target_pose.orientation,
            camera_extrinsics=wrist_camera_extrinsics,
            camera_intrinsics=wrist_camera_intrinsics,
            camera_depth=wrist_camera_depth,
            image_width=self.image_size,
            image_height=self.image_size,
            tolerance=tolerance,
            step_width=0.003
        )
        
        # print("wst vis:", vis)
        
        
        # draw a shooting line from gripper to the stopping point
        center = (u_target, v_target)
        
        if points_uv:
            spans = self._find_span(vis, min_length=1)
            if spans:
                if len(spans) > 1 and spans[1][1]-spans[1][0] > spans[0][1]-spans[0][0]+ 8:
                    _, end_id = spans[1]
                else:
                    _, end_id = spans[0]
                center = points_uv[end_id]
                Z_c = wrist_camera_depth[center[1], center[0]]

        # if visibility is -3, still should draw the reticle
        target_orientation = R.from_quat(target_pose.orientation).as_matrix()
        R_camera = wrist_camera_extrinsics[:3, :3] @ target_orientation
        theta = np.arctan2(R_camera[1, 0], R_camera[0, 0])
        # caculate the angle rotation at the wrist camera frame

        # Define line length adaptively based on the distance to the camera
        line_length = max((0.7 - Z_c)*50, 0) + 20
        
        # thickness = 1 if line_length < 30 else 2
        thickness = 1

        if line_length > 50:
            center_size = 3
        elif line_length < 30:
            center_size = 1
        else:
            center_size = 2

        # Compute horizontal and vertical reticle lines
        horiz_right_start_uv, horiz_right_end_uv = self.get_rotated_line(center, theta, line_length)
        wrist_camera_rgb = cv2.line(
            wrist_camera_rgb, horiz_right_start_uv, horiz_right_end_uv, self.wrscam_reticle_color, thickness
        )

        horiz_left_start_uv, horiz_left_end_uv = self.get_rotated_line(center, theta + np.pi, line_length)
        wrist_camera_rgb = cv2.line(
            wrist_camera_rgb, horiz_left_start_uv, horiz_left_end_uv, self.wrscam_reticle_color, thickness
        )

        vert_up_start_uv, vert_up_end_uv = self.get_rotated_line(center, theta + np.pi / 2, line_length)
        wrist_camera_rgb = cv2.line(
            wrist_camera_rgb, vert_up_start_uv, vert_up_end_uv, self.wrscam_reticle_color, thickness
        )

        vert_down_start_uv, vert_down_end_uv = self.get_rotated_line(center, theta - np.pi / 2, line_length)
        wrist_camera_rgb = cv2.line(
            wrist_camera_rgb, vert_down_start_uv, vert_down_end_uv, self.wrscam_reticle_color, thickness
        )

        # plot a center point on the image using cv2
        center_color = self.wrscam_center_color_open if target_pose.gripper_open else self.wrscam_center_color_close
        wrist_camera_rgb = cv2.circle(wrist_camera_rgb, center, center_size, center_color, -1)

        if self.use_waypoint:
            # add a line from the current gripper pose to the next pose
            cur_u, cur_v, _ = self._3d_to_2d(gripper_pose.position, wrist_camera_extrinsics, wrist_camera_intrinsics)
            wrist_camera_rgb = cv2.line(
                wrist_camera_rgb, (cur_u, cur_v), (u_target, v_target), self.wrscam_line_color_ee2target, 1
            )

            # draw a small circle on the current gripper pose
            wrist_camera_rgb = cv2.circle(
                wrist_camera_rgb, (cur_u, cur_v), center_size, self.wrscam_line_color_ee2target, -1
            )

        self.last_center = center
        self.last_Zh = Z_c
        return wrist_camera_rgb, visibility



class VideoRecorder:
    def __init__(self, num_image, output_dir, fps=10):
        assert num_image in [1, 2, 4]
        self.output_dir = output_dir
        self.fps = fps
        self.video_writer = None

    def start(self, video_name):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        video_path = os.path.join(self.output_dir, video_name)
        if "mp4" not in video_path:
            video_path += ".mp4"
        if os.path.exists(video_path):
            os.remove(video_path)
        self.video_writer = imageio.get_writer(video_path, fps=self.fps)

    def add_frame(self, frame: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(frame, list): # concatenate multiple images
            if len(frame) == 2:
                frame = np.concatenate(frame, axis=1)
            elif len(frame) == 4:
                frame = np.concatenate([np.concatenate(frame[:2], axis=1), np.concatenate(frame[2:], axis=1)], axis=0)
            else:
                raise ValueError("Only support 2 or 4 images to be concatenated")
        self.video_writer.append_data(frame)

    def close(self):
        self.video_writer.close()