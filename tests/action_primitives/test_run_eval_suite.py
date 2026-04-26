"""Unit tests for the pure logic in run_eval_suite — run-list construction
and the per-eval cmd builder. The HF download + subprocess + upload paths
are end-to-end-tested by actually running the suite."""
from pathlib import Path

from experiments.action_primitives.run_eval_suite import (
    DEFAULT_PROBES,
    DEFAULT_SLICES,
    EvalRun,
    PROBE_SLICE,
    _build_eval_cmd,
    build_runs,
)


def test_default_run_list_has_5_slices_plus_3_probes():
    runs = build_runs()
    assert len(runs) == len(DEFAULT_SLICES) + len(DEFAULT_PROBES)


def test_default_run_list_first_section_is_slices_with_none_probe():
    runs = build_runs()
    for i, slice_name in enumerate(DEFAULT_SLICES):
        assert runs[i].slice_name == slice_name
        assert runs[i].probe == "none"
        assert runs[i].json_filename == f"eval_{slice_name}_none.json"


def test_default_run_list_second_section_is_probes_on_probe_slice():
    runs = build_runs()
    probe_runs = runs[len(DEFAULT_SLICES):]
    for run, probe in zip(probe_runs, DEFAULT_PROBES, strict=True):
        assert run.slice_name == PROBE_SLICE
        assert run.probe == probe
        assert run.json_filename == f"eval_{PROBE_SLICE}_{probe}.json"


def test_custom_slices_and_probes_override_defaults():
    runs = build_runs(slices=("custom_a", "custom_b"), probes=("p1",), probe_slice="custom_b")
    assert len(runs) == 3
    assert runs[0].slice_name == "custom_a"
    assert runs[0].probe == "none"
    assert runs[2].slice_name == "custom_b"
    assert runs[2].probe == "p1"


def test_eval_cmd_contains_all_required_args():
    run = EvalRun("phase_a_holdout", "none", "eval_phase_a_holdout_none.json")
    cmd = _build_eval_cmd(
        run=run,
        checkpoint_path="/tmp/ckpt/best.pt",
        data_dir="/tmp/data",
        json_out=Path("/tmp/out.json"),
        device="cuda",
        phase="b0",
        n_rollouts=200,
        decode="expected",
    )
    # Required args from the user's original bash loop must all be present
    for required in ["--checkpoint", "--data-dir", "--n-rollouts", "--device",
                     "--decode", "--skip-offline", "--phase", "--slice",
                     "--report-by-tier", "--json-out"]:
        assert required in cmd, f"missing {required} from generated cmd"
    # Probe=none means --instruction-probe must NOT appear (default behavior)
    assert "--instruction-probe" not in cmd


def test_eval_cmd_appends_instruction_probe_for_non_none_probe():
    run = EvalRun("multi_btn_generic", "shuffled", "eval_multi_btn_generic_shuffled.json")
    cmd = _build_eval_cmd(
        run=run,
        checkpoint_path="/tmp/ckpt/best.pt",
        data_dir="/tmp/data",
        json_out=Path("/tmp/out.json"),
        device="cuda",
        phase="b0",
        n_rollouts=200,
        decode="expected",
    )
    assert "--instruction-probe" in cmd
    assert cmd[cmd.index("--instruction-probe") + 1] == "shuffled"


def test_eval_cmd_runs_evaluate_module():
    run = EvalRun("phase_a_holdout", "none", "x.json")
    cmd = _build_eval_cmd(
        run=run,
        checkpoint_path="/c",
        data_dir="/d",
        json_out=Path("/o"),
        device="cuda",
        phase="b0",
        n_rollouts=1,
        decode="expected",
    )
    # Must invoke the evaluate module (not train, not anything else)
    assert "experiments.action_primitives.evaluate" in cmd
    assert "-m" in cmd
