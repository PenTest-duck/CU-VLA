"""Unit test for the step-ckpt pruning logic in train.py.

The pruning is inlined inside train.py's main loop (no helper function exposed),
so we mirror it here and assert the behavior contract: keep the latest N
step_*.pt files, delete older ones, never touch best.pt or final.pt.

If train.py's pruning implementation diverges from this mirror, update the
mirror to match.
"""
from pathlib import Path


KEEP_LAST_N_STEP_CKPTS = 3  # must match train.py:KEEP_LAST_N_STEP_CKPTS


def _prune(out_dir: Path) -> list[str]:
    """Mirror of the pruning block in train.py. Returns list of pruned filenames
    (for assertions); production code just unlinks + prints.
    """
    pruned = []
    existing_step_ckpts = sorted(out_dir.glob("step_*.pt"))
    for old in existing_step_ckpts[:-KEEP_LAST_N_STEP_CKPTS]:
        pruned.append(old.name)
        old.unlink()
    return pruned


def _make_ckpt(path: Path) -> None:
    path.write_bytes(b"fake-ckpt")


def test_keeps_latest_n_when_exactly_n_plus_one_exist(tmp_path):
    for n in [10, 20, 30, 40]:
        _make_ckpt(tmp_path / f"step_{n:05d}.pt")
    pruned = _prune(tmp_path)
    assert pruned == ["step_00010.pt"]
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_00020.pt", "step_00030.pt", "step_00040.pt"]


def test_no_prune_when_under_keep_threshold(tmp_path):
    # Only 2 ckpts exist (under N=3); nothing should be pruned.
    _make_ckpt(tmp_path / "step_00010.pt")
    _make_ckpt(tmp_path / "step_00020.pt")
    pruned = _prune(tmp_path)
    assert pruned == []
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_00010.pt", "step_00020.pt"]


def test_prune_many_keeps_only_latest_n(tmp_path):
    for n in [10, 20, 30, 40, 50, 60, 70]:
        _make_ckpt(tmp_path / f"step_{n:05d}.pt")
    pruned = _prune(tmp_path)
    # Latest 3 kept = 50, 60, 70 → pruned = 10, 20, 30, 40
    assert pruned == ["step_00010.pt", "step_00020.pt", "step_00030.pt", "step_00040.pt"]
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_00050.pt", "step_00060.pt", "step_00070.pt"]


def test_best_pt_and_final_pt_never_pruned(tmp_path):
    # best.pt + final.pt sit alongside several step ckpts; pruning must NOT
    # touch them even though they're older than the kept step ckpts.
    _make_ckpt(tmp_path / "best.pt")
    _make_ckpt(tmp_path / "final.pt")
    for n in [10, 20, 30, 40, 50]:
        _make_ckpt(tmp_path / f"step_{n:05d}.pt")
    pruned = _prune(tmp_path)
    assert pruned == ["step_00010.pt", "step_00020.pt"]
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "final.pt").exists()


def test_lex_sort_is_numeric_for_5_digit_padding(tmp_path):
    # Sanity: 5-digit zero-padding means lex sort == numeric sort for the
    # values train.py actually emits. step_00099.pt < step_00100.pt lexically
    # AND numerically, so sorted() picks the right "latest".
    for n in [99, 100, 5, 50]:
        _make_ckpt(tmp_path / f"step_{n:05d}.pt")
    pruned = _prune(tmp_path)
    assert pruned == ["step_00005.pt"]
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_00050.pt", "step_00099.pt", "step_00100.pt"]
