# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lelab.record — request schemas and handler entry points."""

from __future__ import annotations

import pytest


def test_recording_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from lelab.record import RecordingRequest

    with pytest.raises(ValidationError):
        RecordingRequest()


def test_recording_status_handler_exposes_state_fields() -> None:
    from lelab.record import handle_recording_status

    result = handle_recording_status()
    assert isinstance(result, dict)
    # Pinning the exact keys so a rename in handle_recording_status surfaces here.
    assert "recording_active" in result
    assert "current_phase" in result
    assert "session_ended" in result
    assert "available_controls" in result


def test_handle_stop_recording_when_idle_returns_dict(tmp_lerobot_home) -> None:
    from lelab.record import handle_stop_recording

    result = handle_stop_recording()
    assert isinstance(result, dict)


def test_create_record_config_pins_dshow_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, recording must use the DSHOW backend so a camera_index opens
    the same device /available-cameras enumerated (via pygrabber, DSHOW order).
    """
    import lelab.record as record
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}},
    )

    config = record.create_record_config(request)
    assert config.robot.cameras["wrist"].backend == Cv2Backends.DSHOW


def test_build_camera_configs_uses_default_backend_when_unset() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.AVFOUNDATION
    assert configs["cam"].fourcc is None
    assert configs["cam"].index_or_path == 0


def test_build_camera_configs_passes_fourcc_through() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "fourcc": "MJPG"}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs["cam"].fourcc == "MJPG"


def test_build_camera_configs_explicit_backend_overrides_default() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "V4L2"}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.V4L2


def test_build_camera_configs_invalid_backend_raises() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "NOPE"}}
    with pytest.raises(KeyError):
        _build_camera_configs(cameras, Cv2Backends.ANY)


def test_build_camera_configs_skips_non_opencv_type() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "realsense", "camera_index": 0}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs == {}


def test_create_record_config_sets_explicit_local_root_when_resuming(
    tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lelab.record as record
    import lerobot.utils.constants as constants

    monkeypatch.setattr(constants, "HF_LEROBOT_HOME", tmp_lerobot_home)
    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/existing_dataset",
        single_task="pick up the cube",
        resume=True,
    )

    config = record.create_record_config(request)

    assert config.resume is True
    assert config.dataset.root == tmp_lerobot_home / "user" / "existing_dataset"


def test_local_dataset_root_rejects_path_traversal(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.record as record
    import lerobot.utils.constants as constants

    monkeypatch.setattr(constants, "HF_LEROBOT_HOME", tmp_lerobot_home)

    with pytest.raises(ValueError, match="Invalid dataset id"):
        record._local_dataset_root("../outside")


def test_resume_requires_an_existing_local_dataset(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch) -> None:
    import lelab.record as record
    import lerobot.utils.constants as constants
    from lelab import rollout, teleoperate

    monkeypatch.setattr(constants, "HF_LEROBOT_HOME", tmp_lerobot_home)
    monkeypatch.setattr(record, "recording_active", False)
    monkeypatch.setattr(teleoperate, "teleoperation_active", False)
    monkeypatch.setattr(rollout, "inference_active", False)

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/missing",
        single_task="pick up the cube",
        resume=True,
    )

    result = record.handle_start_recording(request)

    assert result["success"] is False
    assert "not available locally" in result["message"]
    assert record.recording_active is False


def test_existing_dataset_can_append_an_episode(tmp_lerobot_home) -> None:
    import numpy as np

    from lerobot.datasets import LeRobotDataset

    repo_id = "user/resumable"
    root = tmp_lerobot_home / "user" / "resumable"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (1,), "names": None},
    }

    dataset = LeRobotDataset.create(repo_id, fps=30, root=root, features=features, use_videos=False)
    dataset.add_frame(
        {
            "observation.state": np.array([0], dtype=np.float32),
            "action": np.array([0], dtype=np.float32),
            "task": "pick up the cube",
        }
    )
    dataset.save_episode()
    dataset.finalize()

    resumed = LeRobotDataset.resume(repo_id, root=root)
    resumed.add_frame(
        {
            "observation.state": np.array([1], dtype=np.float32),
            "action": np.array([1], dtype=np.float32),
            "task": "pick up the cube",
        }
    )
    resumed.save_episode()
    resumed.finalize()

    reopened = LeRobotDataset(repo_id, root=root)
    assert reopened.num_episodes == 2
    assert reopened.num_frames == 2
