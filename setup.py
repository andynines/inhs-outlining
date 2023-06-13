import sys
import subprocess

def run_command(command):
    subprocess.run(command, shell=True, check=True)

# Create virtual environment
venv_dir = "venv"

run_command(f"python -m venv {venv_dir}")

if sys.platform == "win32":
    activate_cmd = f"{venv_dir}\\Scripts\\activate.bat"
else:
    activate_cmd = f"source {venv_dir}/bin/activate"

# Activate virtual environment
run_command(activate_cmd)

# Install dependencies
run_command(f"pip install -r requirements.txt")

# Create Jupyter kernel
kernel_name = "inhs-outlining-venv-kernel"
kernel_command = f"{sys.executable} -m ipykernel install --user --name={kernel_name}"

run_command(kernel_command)

print(f"""
Setup complete. Virtual environment created in '{venv_dir}' and Jupyter kernel '{kernel_name}' installed.

To use this virtual environment in a notebook, select the '{kernel_name}' kernel from the 'Kernel' menu.
""")
