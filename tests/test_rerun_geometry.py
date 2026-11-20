"""Check mesh dimensions and linkage closure against the default engine."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from reachy_mini_rust_kinematics import ReachyMiniRustKinematics
from rerun.urdf import UrdfTree

from expressive_ai.rerun.reachy_view import DEFAULT_ASSETS, prepare_model, origin_matrix, log_state


class Capture:
    def __init__(self):
        self.rods = {}

    def log(self, path, value):
        if path.startswith('transforms/rod_'):
            rotation = np.array(value.mat3x3.as_arrow_array().to_pylist()).reshape(3, 3).T
            translation = np.array(value.translation.as_arrow_array().to_pylist()).reshape(3)
            self.rods[int(path.rsplit('_', 1)[1])] = (rotation, translation)


class GeometryTest(unittest.TestCase):
    def test_rods_are_rigid_and_close_at_head(self):
        source = DEFAULT_ASSETS
        config = json.loads((source / 'kinematics_data.json').read_text())
        engine = ReachyMiniRustKinematics(config['motor_arm_length'], config['rod_length'])
        for motor in config['motors']:
            engine.add_branch(motor['branch_position'], np.linalg.inv(motor['T_motor_world']), 1 if motor['solution'] else -1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.urdf'
            geometry = prepare_model(source / 'robot_no_collision.urdf', path)
            model = UrdfTree.from_file_path(path)
            head_inverse, joints, rods = geometry
            for yaw, height, pitch in [(0, 0, 0), (.3, .01, .1), (-.3, -.01, -.1)]:
                with self.subTest(yaw=yaw):
                    pose = np.eye(4)
                    c, s = np.cos(pitch), np.sin(pitch)
                    pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
                    pose[2, 3] = height + config['head_z_offset']
                    angles = engine.inverse_kinematics(pose.tolist(), yaw)
                    head_world = pose @ head_inverse
                    pose[2, 3] -= config['head_z_offset']
                    capture = Capture()
                    log_state(capture, model, {'head_joints': [yaw] + list(angles), 'head_pose': {'m': pose.ravel().tolist()}}, geometry, config['head_z_offset'])
                    self.assertEqual(len(capture.rods), 6)
                    for i, (rotation, translation) in capture.rods.items():
                        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                        self.assertAlmostEqual(np.linalg.det(rotation), 1, places=6)
                        local_end = rods[i][1]
                        attachment = origin_matrix(joints[f'closing_{i}_2_frame'])[:, 3] if i < 6 else np.array([0, 0, 0, 1])
                        expected = (head_world @ attachment)[:3]
                        np.testing.assert_allclose(translation + rotation @ local_end, expected, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
