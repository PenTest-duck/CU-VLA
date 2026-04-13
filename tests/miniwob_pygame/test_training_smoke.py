import os
import tempfile

import numpy as np
import pytest
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


FEATURES = Features({
    "episode_id": Value("int32"),
    "timestep": Value("int32"),
    "image": Image(),
    "cursor_x": Value("float32"),
    "cursor_y": Value("float32"),
    "action_dx": Value("float32"),
    "action_dy": Value("float32"),
    "action_mouse_left": Value("int8"),
    "action_keys_held": Sequence(Value("int8"), length=43),
    "episode_length": Value("int32"),
    "task_name": Value("string"),
    "success": Value("bool"),
})


def _make_synthetic_rows(task: str, num_episodes: int = 4, steps_per_ep: int = 20):
    """Generate synthetic dataset rows for testing."""
    rng = np.random.default_rng(42)
    rows = []
    for ep in range(num_episodes):
        for t in range(steps_per_ep):
            rows.append({
                "episode_id": ep,
                "timestep": t,
                "image": PILImage.fromarray(
                    rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
                ),
                "cursor_x": float(rng.random()),
                "cursor_y": float(rng.random()),
                "action_dx": float(rng.standard_normal()),
                "action_dy": float(rng.standard_normal()),
                "action_mouse_left": int(rng.integers(0, 2)),
                "action_keys_held": [int(x) for x in rng.integers(0, 2, 43)],
                "episode_length": steps_per_ep,
                "task_name": task,
                "success": True,
            })
    return rows


@pytest.fixture
def synthetic_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["click-target", "type-field"]:
            rows = _make_synthetic_rows(task)
            ds = Dataset.from_list(rows, features=FEATURES)
            task_dir = os.path.join(tmpdir, task)
            ds.save_to_disk(task_dir)
        yield tmpdir


def test_training_one_epoch(synthetic_data_dir):
    from experiments.miniwob_pygame.train import train

    train(
        backbone="resnet18",
        chunk_size=5,
        batch_size=4,
        data_dir=synthetic_data_dir,
        device="cpu",
        max_epochs=1,
    )
