import logging

import torch
from wenet.llm_asr.llmasr_model_instruct_version import LLMASR_Model as LLMASR_Model_Instruct
# from wenet.llm_asr.llmasr_model_base_version import LLMASR_Model as LLMASR_Model_Base
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules
from wenet.utils.cmvn import load_cmvn

from gxl_ai_utils.utils import utils_file


def init_llmasr(args, configs, is_inference=False):
    llm_path = configs["llm_path"]
    lora = configs["use_lora"]
    lora_alpha = configs["lora_alpha"]
    lora_rank = configs["lora_rank"]
    lora_dropout = configs["lora_dropout"]

    if configs['encoder'] == 'transformer':
        encoder_type = configs.get('encoder', 'conformer')
        input_dim = configs['input_dim']
        from wenet.utils.init_model import WENET_ENCODER_CLASSES
        encoder = WENET_ENCODER_CLASSES[encoder_type](
            input_dim,
            global_cmvn=None,
            **configs['encoder_conf'],
            **configs['encoder_conf']['efficient_conf']
            if 'efficient_conf' in configs['encoder_conf'] else {})
        encoder_output_dim = configs['encoder_conf']['output_size']
    elif configs['encoder'] == 'whisper':
        raise NotImplementedError('openai-whisper 还没实现')
    elif configs['encoder'] == 'hubert':
        raise NotImplementedError('hubert 还没实现')
    else:
        encoder_output_dim=0
        encoder = None

    speech_token_num = configs.get('speech_token_num', 0)
    train_speech_out = speech_token_num != 0

    if_instruct = configs.get('if_instruct', False)
    BIGMODEL = LLMASR_Model_Instruct
    model = BIGMODEL(
        encoder=encoder,
        encoder_output_dim=encoder_output_dim,
        llm_path=llm_path,
        lora=lora,
        lora_alpha=lora_alpha,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        is_inference=is_inference,
        downsample_rate=configs.get('downsample_rate',1),
        adapter_type=configs.get('adapter_type', 'osum_echat'),
        speech_token_num=speech_token_num,
        train_speech_out=train_speech_out,
    )

    utils_file.logging_info("init_llmasr()：模型初始化完毕,开始打印模型参数量")
    utils_file.logging_info(f'encoder')
    utils_file.print_model_size(model.encoder)
    utils_file.logging_info(f'llm_model')
    utils_file.print_model_size(model.llama_model)
    utils_file.logging_info(f'speech_transformer')
    utils_file.print_model_size(model.speech_transformer)
    utils_file.logging_info(f'speech_llama_proj')
    utils_file.print_model_size(model.speech_llama_proj)
    utils_file.logging_info(f'speech_head')
    utils_file.print_model_size(model.speech_head)
    

    # logging.info(f'OSUM-EChat：init_llmasr()：开始加载encoder参数，仅仅为了消融2，一会马上删了该逻辑')
    encoder_path = "only_encder_ckpt.pt"
    load_checkpoint(model, encoder_path)
    logging.info(f'OSUM-EChat：init_llmasr()：加载encoder参数完毕')


    logging.info(f'OSUM-EChat：init_salmonn()：开始加载初始化模型')
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        logging.info(f'OSUM-EChat： 设置了初始化模型位置，开始加载，参数文件位置：{args.checkpoint}')
        infos = load_checkpoint(model, args.checkpoint)
    elif hasattr(args, 'checkpoint') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}

    if configs.get('init_step', False):
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    logging.info('OSUM-EChat：加载初始化模型完毕')
    # model.to(torch.float32)

    # logging.info('OSUM-EChat：开始加载instruct LLM模型')
    # load_checkpoint(model.llama_model.model, "/mnt/sfs/asr/env/.cache/transformers/models--Qwen--Qwen2.5-7B-Instruct-1M/llama_model.pt")
    # logging.info('OSUM-EChat：加载instruct LLM模型完毕')


    logging.info('OSUM-EChat：开始选择性冻结模块')
    fire_module = configs.get("fire_module", None)
    if fire_module is None:
        logging.info('OSUM-EChat：没有选择解冻的模块,也就是没有训练参数，直接报错返回')
        raise ValueError('没有选择解冻的模块,也就是没有训练参数，直接报错返回')
    for k, p in model.named_parameters():
        # if k.startswith("llama_model") or k.startswith("speech_encoder"):
        # if k.startswith("llama_model") or k.startswith("speech_transformer"):
        if fire_module == 'link':
            # link 包括下采样块， transformer块， 前后linear块
            if k.startswith("llama_model") or k.startswith("encoder"):
                p.requires_grad = False
        elif fire_module == 'encoder':
            if not k.startswith("encoder"):
                p.requires_grad = False
        elif fire_module == 'llm':
            if not k.startswith("llama_model"):
                p.requires_grad = False
        elif fire_module == 'link_and_encoder':
            # 这里和speech token相关的层不会被冻结
            if k.startswith("llama_model"):
                p.requires_grad = False
        elif fire_module == "link_and_encoder_and_lora":
            pass
        elif fire_module == "link_and_lora":
            if k.startswith("encoder"):
                p.requires_grad = False
        logging.info(f"{k} {p.requires_grad} {p.shape} {p.dtype}")
    logging.info('OSUM-EChat：冻结完毕')
    logging.info(configs)
    logging.info(f"✅ 盘古7B模型已加载 = {configs['llm_path']}")
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # ========== 核心：打印NPU基础信息（极简版，鲁棒性优化） ==========
    logging.info("\n===== NPU/设备基础信息 =====")
    try:
        import torch_npu
        # 兼容假包/真实NPU环境，避免属性不存在报错
        npu_available = getattr(torch_npu.npu, 'is_available', lambda: False)()
        npu_count = getattr(torch_npu.npu, 'device_count', lambda: 0)()
        logging.info(f"✅ torch_npu模块已加载")
        logging.info(f"  - NPU是否可用: {npu_available} ")
        logging.info(f"  - NPU设备数量: {npu_count}")
        npu_version = getattr(torch_npu, '__version__', '假包（无真实版本）')
        logging.info(f"✅ torch_npu 版本: {npu_version}")
        logging.info("📌 获取NPU硬件信息:")
        import subprocess
        # logging.info("\n📌 npu-smi info 原始输出:")
        #     # 确保捕获所有输出（包括stdout/stderr），超时兜底
        # npu_smi_output = subprocess.check_output(
        #         ["npu-smi", "info"], encoding="utf-8",stderr=subprocess.STDOUT,timeout=10  # 防止卡死
        #     )
        #     # 打印原始输出（按行拆分，更清晰）
        # for idx, line in enumerate(npu_smi_output.split("\n")):
        #     if line.strip():  # 只打印非空行
        #         logging.info(f"{line.strip()}")
        #         if idx >= 2:
        #             break  # 只打印前几行，避免日志过长
        logging.info("\n📌 torch_npu API获取的NPU属性:")
        if npu_available:
            for dev_id in range(min(npu_count, 1)):  # 只打印前1个设备，避免刷屏
                try:
                    props = torch_npu.npu.get_device_properties(dev_id)
                    logging.info(f"✅ NPU设备{dev_id}属性: {props}")
                except Exception as e:
                    logging.info(f"  NPU设备{dev_id}属性获取失败: {e}")
        else:
            logging.info("  ❌ torch_npu API无法获取属性（NPU不可用）")
        
    except ImportError:
        logging.info(f"❌ 未检测到torch_npu模块（无NPU环境）")
    
   
    return model, configs
