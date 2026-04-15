# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.0",
#     "datasets>=3.0",
#     "Pillow>=10.0",
#     "huggingface-hub>=0.30",
#     "pygame>=2.6",
# ]
# ///
"""Self-contained data generation script for HuggingFace Jobs.

Clones the repo, installs it, and runs generate_data.py with forwarded args.
CPU-only — no GPU needed.

Launch via:
    uv run python scripts/launch_hf_job.py \
      --flavor cpu-upgrade --timeout 4h --script scripts/hf_job_generate_data.py \
      -- --experiment exp5 -n 10000 --push-to-hub PenTest-duck/cu-vla-exp5-data
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/PenTest-duck/CU-VLA.git"
WORKDIR = "/tmp/cu-vla"

EXPERIMENTS = {
    "exp3": {
        "generate_script": "experiments/miniwob_pygame/generate_data.py",
    },
    "exp5": {
        "generate_script": "experiments/mini_editor/generate_data.py",
    },
}


def main() -> None:
    # Install SDL2 dev libraries (needed by pygame on bare Linux containers)
    subprocess.run(
        ["apt-get", "update", "-qq"],
        check=False, capture_output=True,
    )
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "libsdl2-dev", "libsdl2-image-dev",
         "libsdl2-mixer-dev", "libsdl2-ttf-dev", "libfreetype6-dev"],
        check=False, capture_output=True,
    )
    # Reinstall pygame now that SDL2 is available
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "pygame>=2.6"],
        check=True,
    )

    gen_args = [a for a in sys.argv[1:] if a.strip()]

    experiment = "exp5"  # default to latest
    if "--experiment" in gen_args:
        idx = gen_args.index("--experiment")
        experiment = gen_args[idx + 1]
        gen_args = gen_args[:idx] + gen_args[idx + 2:]

    if experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment}. Options: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    exp = EXPERIMENTS[experiment]

    # Clone repo
    if not os.path.exists(WORKDIR):
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", REPO_URL, WORKDIR], check=True)

    os.chdir(WORKDIR)
    sys.path.insert(0, WORKDIR)

    print(f"Experiment: {experiment}")
    print(f"Running {exp['generate_script']} with args: {gen_args}")

    # Use -m style to ensure proper relative imports
    module = exp["generate_script"].replace("/", ".").removesuffix(".py")
    cmd = [sys.executable, "-m", module] + gen_args
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
