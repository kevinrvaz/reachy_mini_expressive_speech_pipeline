"""Live, read-only Reachy Mini motor telemetry and URDF visualization."""

import copy
import json
import logging
import math
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.urdf import UrdfTree

DEFAULT_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "reachy_mini"


def origin_matrix(joint):
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    r, p, y = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = xyz
    return result


def prepare_model(source, destination, frame_prefix=""):
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename.startswith("file://"):
            mesh.set(
                "filename",
                (source.parent / filename.removeprefix("package://"))
                .resolve()
                .as_uri(),
            )
    joints = {j.get("name"): j for j in root.findall("joint")}
    head_offset = origin_matrix(joints["head_frame"])
    # Attach the head directly to the base: the default engine reports its pose,
    # but does not expose passive ball-joint angles.
    joints["passive_7_z"].find("parent").set("link", "body_foot_3dprint")
    # Each rod is positioned from its two ball-joint attachment points.
    geometry = {}
    for i in range(1, 7):
        rod = joints[f"passive_{i}_z"].find("child").get("link")
        endpoint = joints[f"closing_{i}_1_frame"] if i < 6 else joints["passive_7_x"]
        geometry[i] = (rod, origin_matrix(endpoint)[:3, 3])
        joints[f"passive_{i}_z"].find("parent").set("link", "body_foot_3dprint")
    # Keep geometry calculations in original CAD names; namespace only the file
    # loaded by Rerun so separate robot instances cannot share transform frames.
    output = copy.deepcopy(tree)
    for link in output.getroot().findall("link"):
        link.set("name", frame_prefix + link.get("name"))
    for joint in output.getroot().findall("joint"):
        for tag in ("parent", "child"):
            node = joint.find(tag)
            node.set("link", frame_prefix + node.get("link"))
    output.write(destination)
    return np.linalg.inv(head_offset), joints, geometry


def log_state(rec, model, state, geometry_data, z_offset, frame_prefix=""):
    head_inverse, joints, rods = geometry_data
    head = state.get("head_joints")
    antennas = state.get("antennas_position")
    values = []
    if head is not None:
        if len(head) != 7:
            raise ValueError("Expected seven head/body joint positions")
        values += [("yaw_body", head[0], 1)]
        values += [(f"stewart_{i}", head[i], 1) for i in range(1, 7)]
    if antennas is not None:
        if len(antennas) != 2:
            raise ValueError("Expected two antenna positions")
        values += [("left_antenna", antennas[0], 1), ("right_antenna", antennas[1], 1)]
    for name, angle, sign in values:
        if not math.isfinite(angle):
            raise ValueError(f"Invalid angle for {name}")
        rec.log(f"motors/{name}", rr.Scalars(angle))
        rec.log(
            f"transforms/{name}",
            model.get_joint_by_name(name).compute_transform(sign * angle, clamp=False),
        )
    pose = state.get("head_pose")
    if pose is not None:
        matrix = np.array(pose["m"], dtype=float).reshape(4, 4)
        matrix[2, 3] += z_offset
        matrix = matrix @ head_inverse
        rec.log(
            "transforms/head",
            rr.Transform3D(
                translation=matrix[:3, 3],
                mat3x3=matrix[:3, :3],
                parent_frame=frame_prefix + "body_foot_3dprint",
                child_frame=frame_prefix + "xl_330",
            ),
        )

        if head is not None:
            angles = {name: angle * sign for name, angle, sign in values}
            frames = {"body_foot_3dprint": np.eye(4), "xl_330": matrix}
            by_child = {j.find("child").get("link"): j for j in joints.values()}

            def world(link):
                if link not in frames:
                    joint = by_child[link]
                    transform = origin_matrix(joint)
                    angle = angles.get(joint.get("name"), 0)
                    axis = np.array(
                        [float(v) for v in joint.find("axis").get("xyz").split()]
                    )
                    norm = np.linalg.norm(axis)
                    axis = axis / norm if norm else np.array([1.0, 0.0, 0.0])
                    x, y, z = axis
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    rotation = np.eye(4)
                    rotation[:3, :3] = (
                        np.eye(3)
                        + math.sin(angle) * skew
                        + (1 - math.cos(angle)) * (skew @ skew)
                    )
                    frames[link] = (
                        world(joint.find("parent").get("link")) @ transform @ rotation
                    )
                return frames[link]

            for i, (rod, local_end) in rods.items():
                start = world(f"passive_{i}_link_x")[:3, 3]
                end = (
                    world(f"closing_{i}_2")[:3, 3]
                    if i < 6
                    else (matrix @ origin_matrix(joints["passive_7_z"])[:, 3])[:3]
                )
                direction = end - start
                length = np.linalg.norm(direction)
                if length < 1e-9:
                    continue

                # Align the actual local endpoint vector, including rod 2's
                # diagonal CAD frame. Use only rigid rotations, never scale.
                def basis(vector):
                    x = vector / np.linalg.norm(vector)
                    reference = np.array([0.0, 0.0, 1.0])
                    if abs(x @ reference) > 0.99:
                        reference = np.array([0.0, 1.0, 0.0])
                    y = np.cross(reference, x)
                    y /= np.linalg.norm(y)
                    return np.column_stack((x, y, np.cross(x, y)))

                rotation = basis(direction) @ basis(local_end).T
                rec.log(
                    f"transforms/rod_{i}",
                    rr.Transform3D(
                        translation=start,
                        mat3x3=rotation,
                        parent_frame=frame_prefix + "body_foot_3dprint",
                        child_frame=frame_prefix + rod,
                    ),
                )


class RerunLogHandler(logging.Handler):
    """Send formatted Python messages to a specific recording, including threads."""

    def __init__(self, recording, timeline="time"):
        super().__init__()
        self.recording = recording
        self.timeline = timeline

    def emit(self, record):
        try:
            self.recording.set_time(self.timeline, timestamp=record.created)
            level = "WARN" if record.levelno == logging.WARNING else record.levelname
            self.recording.log(
                "logs/" + record.name.replace(".", "/"),
                rr.TextLog(self.format(record), level=level),
            )
        except Exception:
            self.handleError(record)


@contextmanager
def forward_logs(recording, level=logging.INFO, *, timeline="time"):
    logger = logging.getLogger()
    previous_level = logger.level
    handler = RerunLogHandler(recording, timeline)
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()


def robot_spatial_view(prefix=""):
    root = "/" + prefix.strip("/")
    path = root.rstrip("/")
    return rrb.Spatial3DView(
        name="Reachy Mini",
        origin=root,
        contents=[f"{path}/**", f"-{path}/motors/**", f"-{path}/logs/**"],
        background=rrb.Background(
            kind=rrb.components.BackgroundKind.SolidColor, color=[135, 150, 170, 255]
        ),
        line_grid=False,
    )


def viewer_blueprint(prefix="", *, motors=True, logs=True):
    path = "/" + prefix.strip("/") if prefix.strip("/") else ""
    side = []
    if motors:
        side.append(
            rrb.TimeSeriesView(
                name="Measured motor angles (rad)", origin=f"{path}/motors"
            )
        )
    if logs:
        side.append(rrb.TextLogView(name="Python logs", origin=f"{path}/logs"))
    spatial = robot_spatial_view(prefix)
    return (
        rrb.Horizontal(spatial, rrb.Vertical(*side), column_shares=[2, 1])
        if side
        else spatial
    )


def log_floor(rec, frame_prefix=""):
    """Give the robot a gray floor distinct from the blue-gray background."""
    rec.log(
        "environment/floor",
        # Geometry must explicitly use the named frame of the transform below.
        rr.CoordinateFrame(frame_prefix + "environment/floor"),
        rr.Transform3D(
            translation=[0, 0, -0.002],
            parent_frame=frame_prefix + "body_foot_3dprint",
            child_frame=frame_prefix + "environment/floor",
        ),
        rr.Mesh3D(
            vertex_positions=[
                [-0.35, -0.35, 0],
                [0.35, -0.35, 0],
                [0.35, 0.35, 0],
                [-0.35, 0.35, 0],
            ],
            triangle_indices=[[0, 1, 2], [0, 2, 3]],
            vertex_normals=[[0, 0, 1]] * 4,
            albedo_factor=[190, 190, 190, 255],
        ),
        static=True,
    )


class _ScopedRecording:
    """Prefix entity paths without changing the caller's recording."""

    def __init__(self, recording, prefix):
        self.recording = recording
        self.prefix = prefix

    def log(self, path, *values, **kwargs):
        self.recording.log(f"{self.prefix}/{path}", *values, **kwargs)

    def set_time(self, *args, **kwargs):
        self.recording.set_time(*args, **kwargs)


class ReachyMiniView:
    """Reusable robot visualization in a caller-owned Rerun recording.

    Construction logs static assets, but never spawns a viewer, sends a blueprint,
    or connects to hardware. Call log_state() with your samples, or start() for
    background REST polling. Close this component before closing the recording.
    Timestamp values are Unix seconds on the selected timeline, shared with media.
    Use distinct prefixes for multiple instances in one recording.
    """

    def __init__(
        self,
        recording,
        *,
        prefix="reachy",
        timeline="time",
        urdf=None,
        kinematics=None,
        floor=True,
    ):
        self.prefix = prefix.strip("/")
        if not self.prefix or any(c in self.prefix for c in "*?[]"):
            raise ValueError("prefix must be a nonempty entity path without wildcards")
        if not timeline:
            raise ValueError("timeline must be nonempty")
        self.recording = recording
        self.timeline = timeline
        self._scoped = _ScopedRecording(recording, self.prefix)
        self._frame_prefix = self.prefix + "/"
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._directory = tempfile.TemporaryDirectory(prefix="reachy-rerun-")
        try:
            source = Path(urdf) if urdf else DEFAULT_ASSETS / "robot_no_collision.urdf"
            config = (
                Path(kinematics)
                if kinematics
                else DEFAULT_ASSETS / "kinematics_data.json"
            )
            self._z_offset = json.loads(config.read_text())["head_z_offset"]
            fixed = Path(self._directory.name) / "robot.urdf"
            self._geometry = prepare_model(source, fixed, self._frame_prefix)
            self._model = UrdfTree.from_file_path(fixed, entity_path_prefix=self.prefix)
            recording.log_file_from_path(
                fixed, entity_path_prefix=self.prefix, static=True
            )
            self._scoped.log(
                "",
                rr.CoordinateFrame(self._frame_prefix + "body_foot_3dprint"),
                static=True,
            )
            if floor:
                log_floor(self._scoped, self._frame_prefix)
            recording.flush()
        except BaseException:
            self._directory.cleanup()
            raise

    def blueprint(self, *, motors=True, logs=True):
        """Return a layout to embed in any larger Rerun blueprint."""
        return viewer_blueprint(self.prefix, motors=motors, logs=logs)

    def spatial_view(self):
        """Return only the robot's 3D view."""
        return robot_spatial_view(self.prefix)

    def logging_handler(self, level=logging.INFO):
        """Create a handler to attach to selected application loggers."""
        handler = RerunLogHandler(self._scoped, self.timeline)
        handler.setLevel(level)
        return handler

    def forward_logs(self, level=logging.INFO):
        """Opt in to forwarding root Python logging for this context's duration."""
        return forward_logs(self._scoped, level, timeline=self.timeline)

    def log_state(self, state, *, timestamp=None):
        """Log a daemon-format state; an explicit timestamp synchronizes media."""
        if self._closed:
            raise RuntimeError("ReachyMiniView is closed")
        if timestamp is None:
            timestamp = time.time()
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite Unix seconds")
        self.recording.set_time(self.timeline, timestamp=timestamp)
        log_state(
            self._scoped,
            self._model,
            state,
            self._geometry,
            self._z_offset,
            self._frame_prefix,
        )

    def start(self, *, url="http://localhost:8000", hz=30, timeout=2):
        """Start optional read-only polling in a background thread; return self."""
        if self._closed:
            raise RuntimeError("ReachyMiniView is closed")
        if self._thread is not None:
            raise RuntimeError("Polling is already running; call stop() first")
        if (
            not math.isfinite(hz)
            or hz <= 0
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("hz and timeout must be finite and positive")
        params = {
            "with_control_mode": "false",
            "with_head_pose": "true",
            "with_head_joints": "true",
            "with_body_yaw": "false",
            "with_antenna_positions": "true",
            "use_pose_matrix": "true",
        }
        endpoint = url.rstrip("/") + "/api/state/full?" + urlencode(params)
        self._stop.clear()

        def poll():
            last_warning = -math.inf
            while not self._stop.is_set():
                start = time.monotonic()
                delay = 1 / hz
                try:
                    with urlopen(endpoint, timeout=timeout) as response:
                        state = json.load(response)
                    if not self._stop.is_set():
                        self.log_state(state)
                except (
                    URLError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as error:
                    if start - last_warning > 5:
                        self.recording.set_time(self.timeline, timestamp=time.time())
                        self._scoped.log(
                            "logs/telemetry",
                            rr.TextLog(f"Telemetry unavailable: {error}", level="WARN"),
                        )
                        last_warning = start
                    delay = max(delay, 0.5)
                self._stop.wait(max(0, delay - (time.monotonic() - start)))

        self._thread = threading.Thread(
            target=poll, name=f"{self.prefix}-telemetry", daemon=True
        )
        self._thread.start()
        return self

    def stop(self):
        """Stop polling and wait for the in-flight HTTP request to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def close(self):
        if not self._closed:
            self.stop()
            self._closed = True
            try:
                self.recording.flush()
            finally:
                self._directory.cleanup()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("ReachyMiniView is closed")
        return self

    def __exit__(self, *_):
        self.close()
