"""Print the GPU/software environment and a quick throughput benchmark. Run on the training machine."""
import os
import platform
import sys
import time

print("python", sys.version.split()[0], "|", platform.platform())
print("cpus", os.cpu_count())
try:
    import psutil
    print("ram_gb", round(psutil.virtual_memory().total / 2**30, 1))
except ImportError:
    pass
try:
    import torch
except ImportError:
    sys.exit("torch not installed")
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
if not torch.cuda.is_available():
    sys.exit("CUDA NOT AVAILABLE - check NVIDIA driver / torch CUDA build")
p = torch.cuda.get_device_properties(0)
print("gpu", p.name, "| vram_gb", round(p.total_memory / 2**30, 1), "| sm", f"{p.major}.{p.minor}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); t = time.time()
for _ in range(50):
    y = x @ x
torch.cuda.synchronize()
print("fp16 matmul TFLOPS ~", round(50 * 2 * 4096**3 / (time.time() - t) / 1e12, 1))
conv = torch.nn.Conv1d(32, 64, 9, padding=4).cuda()
z = torch.randn(64, 32, 16000, device="cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    conv(z).sum().backward()
torch.cuda.synchronize()
print("conv1d fwd+bwd ms/iter", round((time.time() - t) / 20 * 1000, 1))
