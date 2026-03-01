import torch
try:
    import torch_npu
except ImportError:
    print("torch_npu not found, please install it first.")

exp_dir = "/home/node1_data0/exp/stage3_asr"
pt_name = "step_832499"
weight_dict = torch.load(f"{exp_dir}/{pt_name}/mp_rank_00_model_states.pt",map_location=torch.device('cpu'))['module']
print(weight_dict.keys())
torch.save(weight_dict, f"{exp_dir}/{pt_name}.pt")
