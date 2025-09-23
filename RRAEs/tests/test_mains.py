import subprocess
import os
import pytest
import shutil

def try_remove(name):
    try:
        shutil.rmtree(name)
    except FileNotFoundError:
        pass

def run_script(script_name):
    try:
        result = subprocess.run(
            ["python", script_name], check=True, capture_output=True, text=True
        )
        try_remove("shift")
        try_remove("folder_name")
        try_remove("2d_gaussian_shift_scale")
        try_remove("gaussian_shift")
        return result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Error running {script_name}:\n{e.stderr}")


@pytest.mark.parametrize(
    "script_name", ["main-MLP.py", "main-CNN.py",  "general-MLP.py", "main-adap-CNN.py", "main-adap-MLP.py", "main-var-CNN.py", "main-CNN1D.py"]
)
def test_scripts(script_name):
    if os.path.exists(script_name):
        output = run_script(script_name)
        assert output is not None
    else:
        pytest.fail(f"Script {script_name} not found")
