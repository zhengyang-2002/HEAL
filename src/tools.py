import torch
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from transformers import LlamaForCausalLM, pipeline, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM

def find_free_gpu():
    free_gpu = None
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)  # 当前已分配的内存
        reserved = torch.cuda.memory_reserved(i)    # 当前保留的内存
        if allocated == 0 and reserved == 0:        # 如果内存使用为 0，说明 GPU 空闲
            free_gpu = i
            break
    return free_gpu


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