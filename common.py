
import onnx
import torch, time
import numpy as np
from onnxsim import simplify
from thop import profile
from collections.abc import Iterable

def setup_seed(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   # 保证每次返回得的卷积算法是确定的

def save_model_to_onnx(model, input, save_path) :
    torch.onnx.export(model, input, save_path, export_params=True, opset_version=12)
    simplified_model, check = simplify(save_path)
    onnx.save_model(simplified_model, save_path)

def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + " TB")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + " GB")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + " MB")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + " KB")
        else:
            clever_nums.append(format % num + " B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums

def benchmark_model(model, input, count=100):
    start_time = time.time()
    memory = 0
    for _ in range(count) :
        torch.cuda.reset_max_memory_allocated()
        if isinstance(input, tuple) :
            model(*input)
        else :
            model(input)
        memory+= torch.cuda.max_memory_allocated()
    if isinstance(input, tuple) :
        flops, params = profile(model, inputs=(*input , ))
    else :
        flops, params = profile(model, inputs=(input , ))
    flops, params = clever_format([flops, params])
    return time.time() - start_time, clever_format(memory/count), flops, params 
