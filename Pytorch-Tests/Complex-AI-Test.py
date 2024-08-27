import torch
import time

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

matrix_size = 32 * 512
x = torch.randn(matrix_size, matrix_size, device=device_cpu)
y = torch.randn(matrix_size, matrix_size, device=device_cpu)

def cpu_test():
    print("--------------------------CPU SPEED----------------------------")
    start = time.time()
    result_cpu = torch.matmul(x, y)
    print(time.time() - start)
    print("Verify device: ", result_cpu.device)

def gpu_test():
    x_gpu = x.to(device_gpu)
    y_gpu = y.to(device_gpu)
    torch.cuda.synchronize()

    for i in range(3):
        print("--------------------------GPU SPEED----------------------------")
        start = time.time()
        result_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        print(time.time() - start)
        print("Verify device: ", result_gpu.device)

def test_exec():
    if torch.cuda.is_available():
        gpu_test()
    else:
        cpu_test()

if __name__ == "__main__":
    test_exec()