import os
import subprocess

from hatchling.plugin import hookimpl


class CudaSourcePlugin:
    @hookimpl
    def hatch_register_environment_source(self):
        try:
            # Try to get CUDA version using nvidia-smi
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
            )
            # If we get here, CUDA is available
            cuda_version = output.decode("utf-8").strip()

            # Map CUDA capability to PyTorch CUDA version
            cuda_map = {
                "8.0": "cu118",
                "8.6": "cu118",
                "8.9": "cu121",
                "9.0": "cu121",
            }

            # Get CUDA version from environment or detect
            cuda_choice = os.environ.get("TORCH_CUDA_VERSION")
            if cuda_choice:
                torch_index = f"https://download.pytorch.org/whl/{cuda_choice}"
            else:
                # Find appropriate CUDA version
                for cap, cu_ver in cuda_map.items():
                    if cuda_version.startswith(cap):
                        torch_index = f"https://download.pytorch.org/whl/{cu_ver}"
                        break
                else:
                    # Default to latest supported version if capability not found
                    torch_index = "https://download.pytorch.org/whl/cu121"

        except (subprocess.CalledProcessError, FileNotFoundError):
            # If nvidia-smi fails or isn't found, return CPU version
            torch_index = "https://download.pytorch.org/whl/cpu"

        return {"pytorch": {"sources": ["https://pypi.org/simple", torch_index]}}
