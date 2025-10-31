import torch, sys, os

src = r"Wav2Lip/checkpoints/Wav2Lip-SD-NOGAN.pt"
dst = r"Wav2Lip/checkpoints/wav2lip_converted.pth"

ckpt = torch.load(src, map_location="cpu")
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    torch.save(ckpt["state_dict"], dst)
    print("已导出 state_dict ->", dst)
else:
    print("该 .pt 看起来是 TorchScript（或未包含 state_dict），无法转 pth。请改用 torch.jit.load 或获取官方 .pth。")