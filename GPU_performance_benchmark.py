import time
import torch

d = 8192
x = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")
y = torch.randn(size=(d, d)).to(torch.bfloat16).to("cuda")

def fun(x):
    for _ in range(50):
        x = x @ y.T
    return x


for _ in range(10):
    fun(x)
    torch.cuda.synchronize()

tic = time.time()
for _ in range(10):
    fun(x)
    torch.cuda.synchronize()
toc = time.time()
s = toc - tic
msec = 1e3 * s
tf = (d**3) * 2 * 50 * 10 / (1024**4)
print(f"{msec=:.3f}")
tflops = tf / s
print(f"{tflops=:.3f}")
