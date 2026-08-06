# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bimanual SO-101 Real-Robot Gr00T Policy Evaluation

Closed-loop policy evaluation on a 12-DoF bimanual SO-101, via the GR00T Policy API.

Bimanual counterpart of ``../SO100/eval_so100.py``, and deliberately the same shape:
connect the robot, then loop observation -> policy -> action chunk -> execute. Two
differences:

  * 12-D state/action split into left_arm / left_gripper / right_arm / right_gripper
    and three cameras, driven through LeRobot's ``bi_so_follower``.
  * Runs at the TRAINING DATASET fps, as the official clients do (DROID hardcodes
    15 Hz for its 15 fps data; SO100 sleeps to 1/30 for its 30 fps data). These
    checkpoints were trained on 10 fps data, so a 16-step chunk covers 1.6 s.

One invocation is one episode: Ctrl-C when it succeeds or stalls, reset the scene,
run again. Success rate is tallied by the operator.

Setup and bring-up: CHEATSHEET.md, section "4c. 真機 eval 拍板".
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import zmq

try:
    import draccus
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollowerConfig
    from lerobot.utils.utils import init_logging, log_say
except ModuleNotFoundError as exc:
    if exc.name is not None and (
        exc.name in {"draccus", "lerobot"} or exc.name.startswith("lerobot.")
    ):
        raise ModuleNotFoundError(
            "Bimanual SO-101 evaluation runs in the LeRobot client environment. "
            "See CHEATSHEET.md section 4c."
        ) from None
    raise

from gr00t.eval._horizon_contract import (  # noqa: E402
    PolicyHorizonSpec,
    migrate_deprecated_action_horizon_argv,
)
from gr00t.policy.server_client import PolicyClient  # noqa: E402

# Must match examples/SO101_bimanual/modality.json.
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CAMERAS = ("center", "wrist_left", "wrist_right")
STATE_KEYS = tuple(f"left_{j}.pos" for j in JOINTS) + tuple(
    f"right_{j}.pos" for j in JOINTS
)
LANG_KEY = "annotation.human.task_description"


class So101BimanualAdapter:
    """Between the robot's flat dicts and GR00T's nested modality dicts."""

    def __init__(self, policy_client: PolicyClient, lang_instruction: str):
        self.policy = policy_client
        self.lang = lang_instruction

    def obs_to_policy_inputs(self, obs: dict[str, Any]) -> dict:
        def arm(prefix: str) -> np.ndarray:
            # float32 is load-bearing: LeRobot returns Python floats and a bare
            # np.array() gives float64, which the server rejects outright.
            return np.asarray(
                [obs[f"{prefix}{j}.pos"] for j in JOINTS], dtype=np.float32
            )

        left, right = arm("left_"), arm("right_")
        return {
            # LeRobot's RealSense backend already yields RGB uint8 HWC -- no cvtColor.
            "video": {k: obs[k][None, None, ...] for k in CAMERAS},
            "state": {
                "left_arm": left[:5].reshape(1, 1, 5),
                "left_gripper": left[5:6].reshape(1, 1, 1),
                "right_arm": right[:5].reshape(1, 1, 5),
                "right_gripper": right[5:6].reshape(1, 1, 1),
            },
            "language": {LANG_KEY: [[self.lang]]},
        }

    @staticmethod
    def decode_action_chunk(chunk: dict, t: int) -> dict[str, float]:
        """One timestep of the chunk as a robot action dict.

        The server has already converted the relative arm actions to absolute joint
        targets using the state we sent, so these go straight to the robot: no adding
        the state back, and no joint remapping.
        """
        left = np.concatenate(
            [chunk["left_arm"][0, t], chunk["left_gripper"][0, t]]
        )
        right = np.concatenate(
            [chunk["right_arm"][0, t], chunk["right_gripper"][0, t]]
        )
        action = {f"left_{j}.pos": float(left[i]) for i, j in enumerate(JOINTS)}
        action.update(
            {f"right_{j}.pos": float(right[i]) for i, j in enumerate(JOINTS)}
        )
        return action

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        chunk, _info = self.policy.get_action(self.obs_to_policy_inputs(obs))
        horizon = chunk["left_arm"].shape[1]
        return [self.decode_action_chunk(chunk, t) for t in range(horizon)]


@dataclass
class EvalConfig:
    """Command-line configuration for bimanual real-robot policy evaluation."""

    policy_host: str = "localhost"
    policy_port: int = 5555
    # Open-loop execution length: how many steps of each predicted chunk are sent to
    # the robot before re-planning. NOT the chunk length itself -- that is the model's
    # `action_horizon`, read back from the server (see PolicyHorizonSpec). `None`
    # executes the full chunk.
    execution_horizon: int | None = None
    lang_instruction: str = "Pick up the vials and place them into the rack"

    # Control rate == training dataset fps (10 fps for these checkpoints). Running at
    # the 30 fps the data was *captured* at would drive the arms 3x too fast.
    fps: float = 10.0

    # Hardware. Defaults match calibration/config/bimanual_so101_record_config.yaml.
    robot_id: str = "bimanual_so101_follower"
    calibration_dir: str = (
        "/home/graphen/sim2real/lerobot/calibration/bimanual_follower"
    )
    left_port: str = "/dev/ttyFollowerLeft"
    right_port: str = "/dev/ttyFollowerRight"
    camera_serials: dict[str, str] = field(
        default_factory=lambda: {
            "wrist_left": "138422071840",
            "center": "138422076217",
            "wrist_right": "138422072598",
        }
    )
    # Per-motor cap on |goal - present| for one write, in degrees (gripper: 0-100 units).
    # At fps=10 a cap of 3.0 means 30 deg/s.
    max_relative_target: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(JOINTS, 3.0) | {"gripper": 8.0}
    )

    play_sounds: bool = False


def make_robot(cfg: EvalConfig) -> BiSOFollower:
    def arm(port: str) -> SOFollowerConfig:
        # use_degrees=True to match how the dataset was recorded.
        return SOFollowerConfig(
            port=port,
            use_degrees=True,
            max_relative_target=dict(cfg.max_relative_target),
        )

    robot = BiSOFollower(
        BiSOFollowerConfig(
            id=cfg.robot_id,
            calibration_dir=Path(cfg.calibration_dir),
            left_arm_config=arm(cfg.left_port),
            right_arm_config=arm(cfg.right_port),
            # All three cameras at top level so observation keys stay unprefixed
            # (`center`/`wrist_left`/`wrist_right`). One under right_arm_config would
            # come out as `right_wrist_right` and no longer match the modality config.
            cameras={
                name: RealSenseCameraConfig(
                    serial_number_or_name=serial, width=640, height=480, fps=30
                )
                for name, serial in cfg.camera_serials.items()
            },
        )
    )
    # calibrate=False: SOFollower.calibrate() blocks on input() if the files are missing.
    try:
        robot.connect(calibrate=False)
    except Exception:
        disconnect(robot)
        raise
    return robot


def disconnect(robot: BiSOFollower) -> None:
    """Per-arm teardown. ``BiSOFollower.disconnect`` requires BOTH arms connected, so
    it raises from a half-connected state and leaves RealSense pipelines open -- which
    aborts the interpreter on the way out."""
    for name in ("left_arm", "right_arm"):
        try:
            arm = getattr(robot, name)
            if arm.is_connected:
                arm.disconnect()
        except Exception:
            logging.exception("%s teardown failed", name)


@draccus.wrap()
def eval(
    cfg: EvalConfig,
):  # noqa: A001 - entry-point name mirrors eval_so100.py
    """One invocation = one episode. Judge the outcome yourself, reset the scene, re-run."""
    init_logging()
    period = 1.0 / cfg.fps

    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    # LINGER=0 on the context, so sockets it creates later cannot block teardown:
    # PolicyClient replaces its socket on timeout without closing the old one, and an
    # orphan holding an undeliverable message makes context.term() hang forever.
    policy_client.context.setsockopt(zmq.LINGER, 0)
    policy = So101BimanualAdapter(policy_client, cfg.lang_instruction)

    # Ask the server for the chunk length instead of assuming it: a bare slice would
    # silently execute fewer steps than requested if the checkpoint predicts a shorter
    # chunk. This also validates execution_horizon against it.
    horizons = PolicyHorizonSpec.from_policy(
        policy_client, n_action_steps=cfg.execution_horizon
    )

    robot = make_robot(cfg)
    log_say(
        f'Policy ready: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )
    logging.info(
        "%.0f Hz, model predicts %d actions per chunk, executing %d of them "
        "-> re-planning every %.2f s. Ctrl-C to stop.",
        cfg.fps,
        horizons.action_horizon,
        horizons.n_action_steps,
        horizons.n_action_steps / cfg.fps,
    )

    try:
        while True:
            obs = robot.get_observation()
            for action in policy.get_action(obs)[: horizons.n_action_steps]:
                tic = time.perf_counter()
                robot.send_action(action)
                elapsed = time.perf_counter() - tic
                if elapsed < period:
                    time.sleep(period - elapsed)
    except KeyboardInterrupt:
        logging.info("stopped")
    finally:
        disconnect(robot)
        policy_client.close()


if __name__ == "__main__":
    if migrate_deprecated_action_horizon_argv():
        # The shared helper emits the tyro-style `--execution-horizon`. This script is
        # draccus, whose flags are underscore-only, so translate before it parses.
        for i, tok in enumerate(sys.argv):
            if tok == "--execution-horizon":
                sys.argv[i] = "--execution_horizon"
            elif tok.startswith("--execution-horizon="):
                sys.argv[i] = "--execution_horizon=" + tok.split("=", 1)[1]
        logging.warning(
            "--action_horizon is deprecated; use --execution_horizon."
        )
    eval()
