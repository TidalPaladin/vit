import torch
import torch.nn.functional as F

x = torch.randn(1, 10, 10, dtype=torch.bfloat16)
w = torch.randn(10, dtype=torch.bfloat16)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    y1 = F.rms_norm(x, x.shape[-1:], weight=w)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
        y2 = F.rms_norm(x, x.shape[-1:], weight=w)


import pdb; pdb.set_trace()