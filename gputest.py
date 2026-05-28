"""Core CUDA/PyTorch readiness check for sm_120-class GPUs."""

import sys

import torch

# ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BLUE = "\033[34m"


def ok(msg: str) -> None:
    print(f"{GREEN}OK:{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL:{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}WARN:{RESET} {msg}")


def info(msg: str) -> None:
    print(f"{BLUE}INFO:{RESET} {msg}")


def explain_problem_context() -> None:
    info(
        "This check was added after CUDA workloads failed on newer GPU "
        "architectures (including sm_120) with 'no kernel image is available "
        "for execution on the device'."
    )
    info(
        "Goal: verify PyTorch can see the GPU, includes the required SM target "
        "in compiled kernels, and can execute a real CUDA kernel."
    )


def check_gpu_stack() -> int:
    print("\n=== PyTorch CUDA Checklist ===\n")
    explain_problem_context()
    print()

    # 1. CUDA availability
    if not torch.cuda.is_available():
        fail("CUDA is not available to PyTorch")
        return 1
    ok("CUDA is available")

    # 2. GPU count
    device_count = torch.cuda.device_count()
    info(f"CUDA device count: {device_count}")
    if device_count == 0:
        fail("No CUDA devices found")
        return 1

    # 3. Inspect first GPU
    device_index = 0
    device_name = torch.cuda.get_device_name(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    required_sm = f"sm_{major}{minor}"

    info(f"GPU name: {device_name}")
    info(f"Compute capability: ({major}, {minor}) -> {required_sm}")

    # 4. PyTorch architectures
    arch_list = torch.cuda.get_arch_list()
    info(f"PyTorch compiled SMs: {arch_list}")
    if required_sm not in arch_list:
        fail(f"PyTorch does NOT support this GPU architecture ({required_sm})")
        warn("Action: install a PyTorch build with matching SM support")
        return 2
    ok("PyTorch supports this GPU architecture")

    # 5. Kernel test
    print("\nRunning CUDA kernel test...")
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x  # noqa: F841
        torch.cuda.synchronize()
    except Exception as exc:
        fail("CUDA kernel execution failed")
        print(exc)
        return 3
    ok("CUDA kernel executed successfully")

    # 6. Final verdict
    print("\n=== FINAL VERDICT ===")
    ok("GPU stack is READY for real workloads\n")
    return 0


def main() -> int:
    return check_gpu_stack()


if __name__ == "__main__":
    sys.exit(main())
