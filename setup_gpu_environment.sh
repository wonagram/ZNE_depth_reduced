#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ZNE GPU environment setup for Ubuntu
# ============================================================
# This script:
#   1. checks Python and NVIDIA GPU availability
#   2. creates .venv
#   3. installs the Python packages used by the ZNE project
#   4. installs the CUDA-compatible Qiskit Aer GPU package
#   5. verifies that Aer can see "GPU"
#
# Run from the ZNE project directory:
#   chmod +x setup_gpu_environment.sh
#   ./setup_gpu_environment.sh
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "============================================================"
echo " ZNE GPU environment setup"
echo "============================================================"
echo

# ---------- Python ----------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: python3 was not found."
    echo "Install Python first, for example:"
    echo "  sudo apt update"
    echo "  sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi

echo "[1/6] Python"
"$PYTHON_BIN" --version

# ---------- NVIDIA GPU / CUDA ----------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "ERROR: nvidia-smi was not found."
    echo "The NVIDIA driver/GPU is not available to this Ubuntu session."
    exit 1
fi

echo
echo "[2/6] NVIDIA GPU"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || nvidia-smi

CUDA_VERSION="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
if [[ -z "$CUDA_VERSION" ]]; then
    echo
    echo "ERROR: Could not determine the CUDA version from nvidia-smi."
    exit 1
fi

CUDA_MAJOR="${CUDA_VERSION%%.*}"
echo "CUDA version reported by NVIDIA driver: $CUDA_VERSION"

if (( CUDA_MAJOR >= 12 )); then
    AER_GPU_PACKAGE="qiskit-aer-gpu"
elif (( CUDA_MAJOR == 11 )); then
    AER_GPU_PACKAGE="qiskit-aer-gpu-cu11"
else
    echo
    echo "ERROR: Qiskit Aer GPU requires CUDA 11.2 or newer."
    echo "Detected CUDA version: $CUDA_VERSION"
    exit 1
fi

echo "Qiskit Aer GPU package selected: $AER_GPU_PACKAGE"

# ---------- Virtual environment ----------
echo
echo "[3/6] Creating virtual environment: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        echo
        echo "ERROR: Could not create the virtual environment."
        echo "On Ubuntu, install the venv package and run this script again:"
        echo "  sudo apt update"
        echo "  sudo apt install -y python3-venv"
        exit 1
    fi
else
    echo "$VENV_DIR already exists; reusing it."
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ---------- Packages ----------
echo
echo "[4/6] Installing ZNE project packages"

# These versions reproduce the supplied working environment as closely as
# practical. qiskit-aer itself is intentionally NOT installed here because
# the CUDA-enabled Aer package is installed separately below.
python -m pip install \
    "annotated-types==0.8.0" \
    "attrs==26.1.0" \
    "certifi==2026.7.22" \
    "cffi==2.1.1" \
    "charset-normalizer==3.5.1" \
    "cirq-core==1.6.1" \
    "contourpy==1.3.3" \
    "cryptography==50.0.1" \
    "cycler==0.12.1" \
    "dill==0.4.1" \
    "duet==0.2.9" \
    "fonttools==4.64.0" \
    "ibm-cloud-sdk-core==3.26.1" \
    "ibm-platform-services==0.77.1" \
    "idna==3.19" \
    "kiwisolver==1.5.1" \
    "matplotlib==3.11.1" \
    "mitiq==1.1.0" \
    "mpmath==1.3.0" \
    "networkx==3.6.1" \
    "numpy==2.2.6" \
    "packaging==26.3" \
    "pandas==2.3.3" \
    "pillow==12.3.0" \
    "ply==3.11" \
    "psutil==7.2.2" \
    "pycparser==3.0" \
    "pydantic==2.13.5" \
    "pydantic_core==2.46.5" \
    "PyJWT==2.13.0" \
    "pylatexenc==2.11" \
    "pyparsing==3.3.2" \
    "pyspnego==0.12.2" \
    "python-dateutil==2.9.0.post0" \
    "pytz==2026.3.post1" \
    "qiskit==2.5.2" \
    "qiskit-ibm-runtime==0.41.1" \
    "requests==2.34.2" \
    "requests_ntlm==1.3.0" \
    "rustworkx==0.18.1" \
    "scipy==1.17.1" \
    "six==1.17.0" \
    "sortedcontainers==2.4.0" \
    "stevedore==5.9.1" \
    "sympy==1.14.0" \
    "tabulate==0.10.0" \
    "tqdm==4.70.0" \
    "typing-inspection==0.4.4" \
    "typing_extensions==4.16.0" \
    "tzdata==2026.3" \
    "urllib3==2.7.0"

# ---------- GPU Aer ----------
echo
echo "[5/6] Installing GPU-enabled Qiskit Aer"

# Remove a CPU-only Aer installation if a dependency happened to install one.
python -m pip uninstall -y qiskit-aer qiskit-aer-gpu qiskit-aer-gpu-cu11 >/dev/null 2>&1 || true
python -m pip install "$AER_GPU_PACKAGE"

# ---------- Verification ----------
echo
echo "[6/6] Verifying Qiskit and GPU Aer"

python - <<'PY'
import sys
import qiskit
import qiskit_aer
from qiskit_aer import AerSimulator

print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("Qiskit version    :", qiskit.__version__)
print("Qiskit Aer version:", qiskit_aer.__version__)

sim = AerSimulator()
devices = tuple(sim.available_devices())
print("Aer devices       :", devices)

if "GPU" not in devices:
    raise SystemExit(
        "\nERROR: Qiskit Aer was installed, but Aer does not report GPU support.\n"
        "Check the NVIDIA driver/CUDA installation and the Aer GPU package."
    )

# Small GPU smoke test.
gpu_sim = AerSimulator(method="statevector", device="GPU")
print("GPU smoke test    : AerSimulator(method='statevector', device='GPU') created successfully")
print("\nSUCCESS: GPU-enabled ZNE Python environment is ready.")
PY

echo
echo "============================================================"
echo " Installation complete"
echo "============================================================"
echo
echo "Activate this environment later with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Then test your batch configuration with:"
echo "  python validate_batch_configuration.py"
echo
echo "Then:"
echo "  python run_experiments.py --dry-run"
echo
echo "And to actually run:"
echo "  python run_experiments.py --run"
echo
