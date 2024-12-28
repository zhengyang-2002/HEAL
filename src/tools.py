import torch
def find_free_gpu():
    free_gpu = None
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)  # 当前已分配的内存
        reserved = torch.cuda.memory_reserved(i)    # 当前保留的内存
        if allocated == 0 and reserved == 0:        # 如果内存使用为 0，说明 GPU 空闲
            free_gpu = i
            break
    return free_gpu