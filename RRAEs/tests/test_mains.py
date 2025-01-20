import subprocess
import os
import pytest
import shutil

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        try:
            shutil.rmtree("shift")
            shutil.rmtree("test_data_CNN")
            shutil.rmtree("folder_name")
        except FileNotFoundError:
            pass
        return result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Error running {script_name}:\n{e.stderr}")

@pytest.mark.parametrize("script_name", ['main-MLP.py', 'main-CNN.py', 'general-MLP.py'])
def test_scripts(script_name):
    if os.path.exists(script_name):
        output = run_script(script_name)
        assert output is not None
    else:
        pytest.fail(f"Script {script_name} not found")