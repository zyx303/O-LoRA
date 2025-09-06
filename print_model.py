import argparse
import torch
import math

def human(n):  # 参数量可读化
    if n < 1e3: return str(n)
    units = ['K','M','B']
    for i,u in enumerate(units,1):
        if n < 1000**(i+1):
            return f"{n/1000**i:.3f}{u}"
    return f"{n:.0f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='full_model.pth', help='state_dict 路径')
    ap.add_argument('--filter', nargs='*', default=[], help='只打印包含这些子串的键')
    ap.add_argument('--summary', action='store_true', help='只打印汇总信息')
    args = ap.parse_args()

    sd = torch.load(args.path, map_location='cpu')
    for k in sd.keys():
        print(k)
    # if not isinstance(sd, (dict,)):
    #     print(f"文件不是字典，类型: {type(sd)}，直接打印：\n{sd}")
    #     return

    # keys = list(sd.keys())
    # print(f"Loaded: {args.path}")
    # print(f"Total keys: {len(keys)}")

    # total_params = 0
    # printed = 0
    # for k, v in sd.items():
    #     if args.filter and not any(f in k for f in args.filter):
    #         continue
    #     if torch.is_tensor(v):
    #         numel = v.numel()
    #         total_params += numel
    #         if not args.summary:
    #             v_min = v.min().item() if v.numel() > 0 else float('nan')
    #             v_max = v.max().item() if v.numel() > 0 else float('nan')
    #             v_norm = v.norm().item() if v.numel() > 0 else float('nan')
    #             print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, numel={numel}, "
    #                   f"min={v_min:.6g}, max={v_max:.6g}, norm={v_norm:.6g}")
    #         printed += 1
    #     else:
    #         if not args.summary:
    #             print(f"{k}: non-tensor type={type(v)} value={v}")
    #         printed += 1

    # print("\nSummary:")
    # print(f"  Printed keys: {printed}")
    # print(f"  Total tensor params: {total_params} ({human(total_params)})")

if __name__ == '__main__':
    main()