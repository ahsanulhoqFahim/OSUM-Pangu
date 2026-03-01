import ast
import logging
import os
import sys

sys.path.insert(0, '../')

import time
import traceback
import torch
import torch_npu
import json  # 新增：用于解析JSON行数据
from common_utils.utils4infer import get_feat_from_wav_path, load_model_and_tokenizer, token_list2wav


is_npu = True
try:
    import torch_npu 
except ImportError:
    is_npu = False
    print("torch_npu is not available. if you want to use npu, please install it.")

gpu_id=0
device = torch.device(f'cuda:{gpu_id}')
# 若使用NPU，自动切换设备（可选优化）
if is_npu:
    device = torch.device(f'npu:{gpu_id}')
checkpoint_path = "step_832499.pt"
config_path = "conf/ct_config_open_pangu.yaml" #记得改

cosyvoice_model_path = "**/CosyVoice-300M-25Hz"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

model, tokenizer = load_model_and_tokenizer(checkpoint_path, config_path, device)
cosyvoice = None  # 暂时不初始化cosyvoice，避免加载模型报错

def do_s2t_speech_understanding(model, input_wav_path, input_prompt):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path, device)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt, cache_implementation="static")[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2T 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text

def do_s2t_chat_no_think(model, input_wav_path, input_prompt):
    """S2T对话（无思考）"""
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path, device)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    
    if is_npu:
        torch_npu.npu.synchronize()
    start_time = time.time()
    
    # ========== 替换为你的真实推理逻辑 ==========
    res_text = model.generate4chat(wavs=feat, wavs_len=feat_lens, prompt=input_prompt,cache_implementation="static")[0]
    # ===========================================
    
    if is_npu:
        torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2T4Chat 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text

def do_s2t_chat_think(model, input_wav_path):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path, device)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate4chat_think(wavs=feat, wavs_len=feat_lens, cache_implementation="static")[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2T4Chat think 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text

def do_t2s(model, text_for_tts):  # 增加 model 参数
    model.eval()
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    res_tensor = model.generate_tts(device=device, text=text_for_tts)[0]
    res_token_list = res_tensor.tolist()
    res_text = res_token_list[:-1]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"T2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text

def do_t2t_chat(model, question_txt):  # 增加 model 参数
    model.eval()
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    print(f'开始t2t推理, question_txt: {question_txt}')
    res_text = model.generate_text2text(device=device, text=question_txt)[0]
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"T2T 推理消耗时间: {end_time - start_time:.2f} 秒")
    return res_text

def do_s2s(model, input_wav_path):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path, device)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    output_text, text_res, speech_res = model.generate_s2s_no_stream_with_repetition_penalty(wavs=feat, wavs_len=feat_lens)
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return f'{output_text[0]}|{str(speech_res[0].tolist()[1:])}'

def do_s2s_think(model, input_wav_path):  # 增加 model 参数
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path, device)
    print(f'feat shape: {feat.shape}, feat_lens: {feat_lens}')
    if is_npu: torch_npu.npu.synchronize()
    start_time = time.time()
    output_text, text_res, speech_res = model.generate_s2s_no_stream_think_with_repetition_penalty(wavs=feat, wavs_len=feat_lens)
    if is_npu: torch_npu.npu.synchronize()
    end_time = time.time()
    print(f"S2S 推理消耗时间: {end_time - start_time:.2f} 秒")
    return f'{output_text[0]}|{str(speech_res[0].tolist()[1:])}'

# 原有：批量推理wav.scp格式文件
def batch_inference_s2t(wav_scp_path, output_txt_path, prompt="你听一下这个录音，然后把答案告诉我 "):
    """
    批量进行S2T推理（支持wav.scp格式：key wav_path）
    
    Args:
        wav_scp_path: wav.scp文件路径
        output_txt_path: 输出结果文件路径
        prompt: 推理使用的提示词
    """
    # 读取wav.scp文件
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    total_files = len(lines)
    processed_files = 0
    
    print(f"开始批量推理（wav.scp格式），共 {total_files} 个音频文件")
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        # 分割key和wav路径
        parts = line.split()
        if len(parts) < 2:
            print(f"第 {i} 行格式错误: {line}")
            continue
            
        key = parts[0]
        wav_path = ' '.join(parts[1:])  # 处理路径中可能有空格的情况
        
        print(f"处理文件 {i}/{total_files}: {key}")
        print(f"音频路径: {wav_path}")
        
        # 检查文件是否存在
        if not os.path.exists(wav_path):
            print(f"警告: 文件不存在 {wav_path}")
            result_text = "ERROR: 文件不存在"
        else:
            try:
                # 进行S2T推理
                result_text = do_s2t_chat_no_think(model, wav_path, prompt)
                processed_files += 1
            except Exception as e:
                print(f"处理文件 {key} 时发生错误: {e}")
                traceback.print_exc()
                result_text = f"ERROR: {str(e)}"
        
        # 保存结果
        results.append(f"{key} {result_text}\n")
        
        # 每处理10个文件输出一次进度
        if i % 10 == 0:
            print(f"已处理 {i}/{total_files} 个文件")
    
    # 写入输出文件（原有逻辑补充：之前遗漏了写入步骤）
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
    print(f"批量推理完成！结果已保存至: {output_txt_path}")
    print(f"成功处理 {processed_files}/{total_files} 个文件")

# 新增：批量推理data.list格式文件（JSON行格式）
def batch_inference_s2t_json_list(json_list_path, output_txt_path, prompt="你听一下这个录音，然后把答案告诉我 "):
    """
    批量进行S2T推理（支持data.list JSON行格式）
    
    Args:
        json_list_path: data.list文件路径（每行一个JSON对象）
        output_txt_path: 输出结果文件路径
        prompt: 推理使用的提示词
    """
    # 读取JSON行文件
    with open(json_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    total_files = len(lines)
    processed_files = 0
    
    print(f"开始批量推理（data.list JSON格式），共 {total_files} 个音频文件")
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            # 解析单行JSON数据
            json_data = json.loads(line)
            
            # 提取key和wav路径（兼容JSON格式）
            key = json_data.get("key", f"unknown_key_{i}")
            wav_path = json_data.get("wav", "")
            
            # 校验wav路径是否存在
            if not wav_path:
                print(f"第 {i} 行缺少 'wav' 字段: {line}")
                result_text = "ERROR: 缺少wav路径"
                results.append(f"{key} {result_text}\n")
                continue
            
            print(f"处理文件 {i}/{total_files}: {key}")
            print(f"音频路径: {wav_path}")
            
            # 检查音频文件是否存在
            if not os.path.exists(wav_path):
                print(f"警告: 文件不存在 {wav_path}")
                result_text = "ERROR: 文件不存在"
            else:
                try:
                    # 进行S2T推理
                    result_text = do_s2t_speech_understanding(model, wav_path, prompt)
                    processed_files += 1
                except Exception as e:
                    print(f"处理文件 {key} 时发生错误: {e}")
                    traceback.print_exc()
                    result_text = f"ERROR: {str(e)}"
            
            # 保存结果（格式：key 推理结果）
            results.append(f"{key} {result_text}\n")
            
            # 每处理10个文件输出一次进度
            if i % 10 == 0:
                print(f"已处理 {i}/{total_files} 个文件")
                
        except json.JSONDecodeError as e:
            print(f"第 {i} 行JSON格式错误: {e}")
            results.append(f"unknown_key_{i} ERROR: JSON格式错误\n")
            continue
        except Exception as e:
            print(f"第 {i} 行处理失败: {e}")
            traceback.print_exc()
            results.append(f"unknown_key_{i} ERROR: {str(e)}\n")
            continue
    
    # 写入输出文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
    
    print(f"JSON格式批量推理完成！结果已保存至: {output_txt_path}")
    print(f"成功处理 {processed_files}/{total_files} 个文件")

# 新增：处理speech-web-questions_en.jsonl格式文件
def batch_inference_llama_questions(jsonl_path, output_json_path, prompt="Please answer the question in the audio"):
    """
    处理speech-web-questions_en.jsonl格式文件，输出指定JSON格式结果
    
    Args:
        jsonl_path: 输入JSONL文件路径
        output_json_path: 输出JSON文件路径
        prompt: 推理使用的英文提示词
    """
    # 读取JSONL文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    total_files = len(lines)
    processed_files = 0
    
    print(f"开始批量推理（speech-web-questions JSONL格式），共 {total_files} 个音频文件")
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            # 解析单行JSON数据
            json_data = json.loads(line)
            
            # 提取关键字段
            question = json_data.get("Questions", f"unknown_question_{i}")
            reference = json_data.get("Answer", "")
            wav_path = json_data.get("WavPath", "")
            
            # 校验必要字段
            if not wav_path:
                print(f"第 {i} 行缺少 'WavPath' 字段")
                prediction = "ERROR: 缺少WavPath字段"
            elif not question:
                print(f"第 {i} 行缺少 'Questions' 字段")
                prediction = "ERROR: 缺少Questions字段"
            else:
                print(f"处理文件 {i}/{total_files}: {question}")
                print(f"音频路径: {wav_path}")
                
                # 检查音频文件是否存在
                if not os.path.exists(wav_path):
                    print(f"警告: 文件不存在 {wav_path}")
                    prediction = "ERROR: 文件不存在"
                else:
                    try:
                        # 进行S2T推理（使用英文提示词）
                        prediction = do_s2t_chat_no_think(model, wav_path, prompt)
                        processed_files += 1
                    except Exception as e:
                        print(f"处理文件 {i} 时发生错误: {e}")
                        traceback.print_exc()
                        prediction = f"ERROR: {str(e)}"
            
            # 构建输出格式
            result_item = {
                "question": question,
                "prediction": prediction,
                "reference": reference
            }
            results.append(result_item)
            
            # 每处理10个文件输出一次进度
            if i % 10 == 0:
                print(f"已处理 {i}/{total_files} 个文件")
                
        except json.JSONDecodeError as e:
            print(f"第 {i} 行JSON格式错误: {e}")
            results.append({
                "question": f"unknown_question_{i}",
                "prediction": "ERROR: JSON格式错误",
                "reference": ""
            })
            continue
        except Exception as e:
            print(f"第 {i} 行处理失败: {e}")
            traceback.print_exc()
            results.append({
                "question": f"unknown_question_{i}",
                "prediction": f"ERROR: {str(e)}",
                "reference": ""
            })
            continue
    
    # 写入输出JSON文件（格式化输出，便于阅读）
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # 如果需要每行一个JSON对象，使用下面的方式：
        # for item in results:
        #     json.dump(item, f, ensure_ascii=False)
        #     f.write('\n')
        
        # 如果需要输出一个JSON数组，使用下面的方式：
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nspeech-web-questions JSONL推理完成！结果已保存至: {output_json_path}")
    print(f"成功处理 {processed_files}/{total_files} 个文件")

# 主程序
if __name__ == "__main__":
    print("开始预热模型...")
    warmup_wav_path = "0001_000026.wav"
    warmup_prompt = "Please answer the question in the audio"  # 改为英文提示词
    print(f"正在预热 ...")
    try:
        # 使用重构后的 do_s2t 函数进行预热，传入对应的模型
        res_text = do_s2t_speech_understanding(model, warmup_wav_path, warmup_prompt)
        print(f'预热完成。ASR推理结果: {res_text}')
    except Exception as e:
        traceback.print_exc()
        print(f"预热时发生错误: {e}")
    
    # ===================== 可配置区域 =====================
    # 切换推理格式：
    # 0: wav.scp格式
    # 1: data.list JSON格式
    # 2: speech-web-questions JSONL格式（新增）
    inference_mode = 2
    
    # 文件路径配置
    json_list_path = "data.list"  # JSON格式文件
    wav_scp_path = "wav.scp"  # wav.scp格式文件
    llama_jsonl_path = "speech-web-questions_en.jsonl"  # 新增：speech-web-questions JSONL文件
    output_txt_path = "infer_speech-web-questions.txt"  # 原输出路径
    output_json_path = "infer_speech-web-questions.json"  # 新增：JSON输出路径
    # =====================================================

    if inference_mode == 2:
        # 推理speech-web-questions JSONL格式文件（新增逻辑）
        if os.path.exists(llama_jsonl_path):
            print("\n开始批量推理（speech-web-questions JSONL格式）...")
            batch_inference_llama_questions(llama_jsonl_path, output_json_path)
        else:
            print(f"警告: speech-web-questions JSONL文件不存在: {llama_jsonl_path}")
            print("将进行单个文件的测试推理...")
            res = do_s2t_speech_understanding(model, warmup_wav_path, warmup_prompt)
            print(f'ASR推理结果: {res}')
            
    elif inference_mode == 1:
        # 推理data.list JSON格式文件
        if os.path.exists(json_list_path):
            print("\n开始批量推理（JSON格式）...")
            batch_inference_s2t_json_list(json_list_path, output_txt_path)
        else:
            print(f"警告: data.list文件不存在: {json_list_path}")
            print("将切换为wav.scp格式推理（若文件存在）")
            if os.path.exists(wav_scp_path):
                batch_inference_s2t(wav_scp_path, output_txt_path)
            else:
                print("wav.scp文件也不存在，进行单个文件测试...")
                res = do_s2t_speech_understanding(model, warmup_wav_path, warmup_prompt)
                print(f'ASR推理结果: {res}')
    else:
        # 推理wav.scp格式文件（原有逻辑）
        if os.path.exists(wav_scp_path):
            print("\n开始批量推理（wav.scp格式）...")
            batch_inference_s2t(wav_scp_path, output_txt_path)
        else:
            print(f"警告: wav.scp文件不存在: {wav_scp_path}")
            print("将进行单个文件的测试推理...")
            res = do_s2t_speech_understanding(model, warmup_wav_path, warmup_prompt)
            print(f'ASR推理结果: {res}')