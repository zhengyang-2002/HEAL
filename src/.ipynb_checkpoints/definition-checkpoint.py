import os
import tools
import torch
import Prompt
from openai import OpenAI
from transformers import LlamaForCausalLM, pipeline, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM

class ModelLoader:
    def __init__(self, model_path: str, lora_path=None, system_prompt='你是一个得力的助手'):
        self.model_path = model_path
        self.lora_path=lora_path
        self.system_prompt=system_prompt
        self.name = os.path.basename(self.model_path)
        
        self.pipeline_instance = self._build_pipeline()
        
    def chat(self, query:str, history=None):
        if history is None:
            history = [{"role": "system", "content": self.system_prompt}]
        history.append({"role": "user", "content": query})
        outputs=self.pipeline_instance(history, max_new_tokens=1024)
        reply=outputs[-1]['generated_text'][-1]['content']
        history.append({"role": "assistant", "content": reply})
        return reply

    def _build_pipeline(self):
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, legacy=False)

        # 查找空闲 GPU
        if (free_gpu := self._find_free_gpu()) is not None:
            device = torch.device(f"cuda:{free_gpu}")  # 指定到空闲的 GPU
            print(f'已分配至空闲GPU：{device}')
        else:
            raise RuntimeError("No free GPU available!")

        # 加载模型
        if 'llama' in self.name.lower():
            model = LlamaForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(device)
        if self.lora_path:
            print(f'正在合并Lora结构，路径:{self.lora_path}')
            model = tools.add_lora(model, self.lora_path)
        model = model.eval()

        # 构建文本生成管道
        pipeline_instance = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            do_sample=True,
            temperature=0.5
        )

        return pipeline_instance

    def _find_free_gpu(self):
        free_gpu = None
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)  
            reserved = torch.cuda.memory_reserved(i)    
            if allocated == 0 and reserved == 0:        
                free_gpu = i
                break
        return free_gpu

class Participant(ModelLoader):
    def __init__(self, model_path: str, initial_rating: int = 1000):
        super().__init__(model_path)  # 调用父类的初始化方法
        self.rating = initial_rating

    def update_rating(self, rating_change: int):
        self.rating += rating_change


class Judge:
    def __init__(self, model_name, base_url, api_key, temperature=0.9):
        self.server=OpenAI(base_url=base_url, api_key=api_key)
        self.temperature=temperature
        self.model_name=model_name

    def score(self, dialogue_a, dialogue_b, score_prompt=Prompt.score_prompt_a):
        message_a=[{"role": "user", "content": score_prompt(dialogue_a)}]
        message_b=[{"role": "user", "content": score_prompt(dialogue_b)}]

        score_a = self.server.chat.completions.create(model=self.model_name, messages=message_a, temperature=self.temperature)
        score_b = self.server.chat.completions.create(model=self.model_name, messages=message_b, temperature=self.temperature)

        return [score_a,score_b]

class EloRatingSystem:
    def __init__(self, k_factor=32):
        self.k_factor=k_factor
        
    def expected_score(self, model_1, model_2):
        return 1/(1+10**((model_2.rating-model_1.rating)/400))

    def update_ratings(self, model_a, model_b, result):
        expected_a=self.expected_score(model_a,model_b)
        expected_b=self.expected_score(model_b,model_a)

        score_change_a=self.k_factor*(result-expected_a)
        score_change_b=self.k_factor*((1-result)-expected_b)

        model_a.update_rating(score_change_a)
        model_b.update_rating(score_change_b)

    def match(self, model_a, model_b, result):
        """
        result:1表示A赢，0.5表示平局，0表示B赢 
        """
        self.update_ratings(model_a, model_b, result)