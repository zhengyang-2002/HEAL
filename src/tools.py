import torch
#from definition import ModelLoader
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from transformers import LlamaForCausalLM, pipeline, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM

def _find_free_gpu():
    free_gpu = None
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)  # 当前已分配的内存
        reserved = torch.cuda.memory_reserved(i)    # 当前保留的内存
        if allocated == 0 and reserved == 0:        # 如果内存使用为 0，说明 GPU 空闲
            free_gpu = i
            break
    return free_gpu


def find_free_gpu(threshold=0.05):
    """
    找到显存使用率少于指定阈值（默认 5%）的 GPU。
    
    参数:
        threshold (float): 显存使用率的阈值，默认为 0.05（5%）。
    
    返回:
        int: 符合条件的 GPU 设备编号，如果没有找到则返回 None。
    """
    for i in range(torch.cuda.device_count()):
        # 获取 GPU 的总显存和当前使用的显存
        total_memory = torch.cuda.get_device_properties(i).total_memory  # GPU 总显存
        allocated_memory = torch.cuda.memory_allocated(i)  # 当前已分配的显存
        reserved_memory = torch.cuda.memory_reserved(i)    # 当前保留的显存
        
        # 计算显存使用率
        print(allocated_memory, reserved_memory, total_memory)
        memory_usage = (allocated_memory + reserved_memory) / total_memory
        
        # 如果显存使用率小于阈值，返回该 GPU 设备编号
        if memory_usage < threshold:
            return i
    
    # 如果没有找到符合条件的 GPU，返回 None
    return None


def merge_lora(base_model_path, lora_path):
    """
    将 LoRA 结构与基座模型合并，并返回合并后的模型。

    参数:
        base_model_path: 基座模型的路径。
        lora_path: LoRA 模型的路径。

    返回:
        合并后的模型。
    """
    # 载入基座模型
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True).half().cuda()
    # 暂存用以验证权重是否改变
    #first_weight = base_model.transformer.encoder.layers[0].self_attention.query_key_value.weight
    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()
    
    # 载入lora结构的模型
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并lora结构
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    
    # 验证结构
    assert not torch.allclose(first_weight_old, first_weight), 'Weight Should Change after Lora Merge'
    
    # 给模型改名
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model.state_dict().items()
        if "lora" not in k
    }
    
    return lora_model



def add_lora(base_model, lora_path):
    """
    将 LoRA 结构添加到基座模型中，并返回合并后的模型。

    参数:
        base_model: 基座模型，已经加载到设备上的模型实例。
        lora_path: LoRA 模型的路径。

    返回:
        合并后的模型。
    """
    # 暂存用以验证权重是否改变
    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()
    
    # 载入 LoRA 结构的模型
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并 LoRA 结构
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    
    # 验证权重是否改变
    assert not torch.allclose(first_weight_old, first_weight), 'Weight Should Change after Lora Merge'
    
    # 返回合并后的模型
    return lora_model

def conduct_dialogue(patient, doctor, rounds: int = 8) -> list:
    """
    让求助者和心理医生进行对话，并记录对话过程。

    :param patient: 扮演求助者的 ModelLoader 实例
    :param doctor: 扮演心理医生的 ModelLoader 实例
    :param rounds: 对话的轮次
    :return: list 记录对话过程的列表，格式为 [<求助者>, <心理医生>, <求助者>, <心理医生>, ...]
    """
    dialogue_history = []  # 用于记录对话过程
    doctor_reply = ""  # 初始输入为空

    for _ in range(rounds):
        # 求助者发言
        patient_query = patient.chat(doctor_reply)
        dialogue_history.append(f"<求助者>{patient_query}")

        # 心理医生回复
        doctor_reply = doctor.chat(patient_query)
        dialogue_history.append(f"<心理医生>{doctor_reply}")

    return dialogue_history


def optimize_contest_list(contest_list):
    # 创建一个字典来存储每个模型对的实验
    model_pair_dict = {}
    
    # 遍历contest_list，将实验按照模型对分组
    for entity in contest_list:
        model_pair = tuple(sorted([entity['model_a'], entity['model_b']]))
        if model_pair not in model_pair_dict:
            model_pair_dict[model_pair] = []
        model_pair_dict[model_pair].append(entity)
    
    # 清空原始的contest_list
    contest_list = []
    
    # 将分组后的实验按照模型对的顺序重新排列
    for model_pair, entities in model_pair_dict.items():
        contest_list.extend(entities)
    
    return contest_list