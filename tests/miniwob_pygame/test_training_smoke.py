import os
import tempfile

import h5py
import numpy as np
import pytest


@pytest.fixture
def synthetic_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for task in ["click-target", "type-field"]:
            task_dir = os.path.join(tmpdir, task, "000")
            os.makedirs(task_dir)
            for ep in range(4):
                path = os.path.join(task_dir, f"episode_{ep:05d}.hdf5")
                T = 20
                with h5py.File(path, "w") as f:
                    f.create_dataset(
                        "observations",
                        data=np.random.randint(0, 255, (T, 224, 224, 3), dtype=np.uint8),
                    )
                    f.create_dataset(
                        "cursor_positions",
                        data=np.random.rand(T, 2).astype(np.float32),
                    )
                    f.create_dataset(
                        "actions_dx",
                        data=np.random.randn(T).astype(np.float32),
                    )
                    f.create_dataset(
                        "actions_dy",
                        data=np.random.randn(T).astype(np.float32),
                    )
                    f.create_dataset(
                        "actions_mouse_left",
                        data=np.random.randint(0, 2, T, dtype=np.int8),
                    )
                    f.create_dataset(
                        "actions_keys_held",
                        data=np.random.randint(0, 2, (T, 43), dtype=np.int8),
                    )
                    f.attrs["task_name"] = task
                    f.attrs["success"] = True
                    f.attrs["num_steps"] = T
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
