from gxl_ai_utils.utils import utils_file
import torch

def do_get_encode_ckpt():
    osum_echat_ckpt_path = "/apdcephfs_qy3/share_976139/users/xuelonggeng/ckpt/osum_echat/models--ASLP-lab--OSUM-EChat/snapshots/d658ae8c15675b8f7ce0ffdee879f99549a1e70b/language_think_final.pt"
    output_encoder_ckpt_path = "/apdcephfs_qy3/share_976139/users/xuelonggeng/ckpt/osum_echat/only_encder_ckpt.pt"
    param_dict = torch.load(osum_echat_ckpt_path, map_location='cpu')
    encoder_dict = {}
    for key in param_dict:
        if key.startswith('encoder.'):
            encoder_dict[key] = param_dict[key]
    torch.save(encoder_dict, output_encoder_ckpt_path)
    print("Encoder ckpt saved to:", output_encoder_ckpt_path)