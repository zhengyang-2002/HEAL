import os
import copy
import tools
import torch
import random
import Prompt
from openai import OpenAI
from transformers import LlamaForCausalLM, pipeline, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM

class ModelLoader:
    def __init__(self, model_path: str, lora_path=None, system_prompt='你是一个杰出的心理治疗师'):
        self.model_path=model_path
        self.lora_path=lora_path
        self.system_prompt=system_prompt
        self.name=os.path.basename(self.model_path)
        self.history=[{"role": "system", "content": self.system_prompt}]
        self.pipeline_instance = self._build_pipeline()

    def clear(self):
        self.history=[{"role": "system", "content": self.system_prompt}]
        
    def chat(self, query:str, history=None):
        if not history:
            history = self.history
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
        self.model=model

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

    def find_free_gpu(self):
        free_gpu = None
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)  
            reserved = torch.cuda.memory_reserved(i)    
            if allocated == 0 and reserved == 0:        
                free_gpu = i
                break
        return free_gpu

    def _find_free_gpu(self, threshold=0.05):
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
            memory_usage = (allocated_memory + reserved_memory) / total_memory
            
            # 如果显存使用率小于阈值，返回该 GPU 设备编号
            if memory_usage < threshold:
                return i

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
    def __init__(self, participants:list, k_factor:int=32, initial_rating:int=1000):
        self.k_factor=k_factor
        self.participants={participant:initial_rating for participant in participants}
        self.scores_log=[copy.deepcopy(self.participants)]
        self.contest_list=[]

    def generate_contest_list(self, rounds):
        temp_list=[]
        self.contest_list=[]
        models=list(self.participants.keys())
        for order in range(rounds):
            candidates=random.sample(models, k=2)
            entity={'model_a':candidates[0], 'model_b':candidates[1], 'dialogue_a':'', 'dialogue_b':'', 'order':order}
            self.contest_list.append(entity)
        self.contest_list=tools.optimize_contest_list(self.contest_list)
        return self.contest_list
        
    def _expected_score(self, model_1, model_2):
        return 1/(1+10**((self.participants[model_2]-self.participants[model_1])/400))

    def _update_ratings(self, model_a, model_b, result):
        expected_a=self._expected_score(model_a,model_b)
        expected_b=self._expected_score(model_b,model_a)

        score_change_a=self.k_factor*(result-expected_a)
        score_change_b=self.k_factor*((1-result)-expected_b)

        self.participants[model_a]+=score_change_a
        self.participants[model_b]+=score_change_b

    def match(self, llm_judge, contest_list):
        """
        win:1表示A赢，0.5表示平局，0表示B赢 
        """

        for entity in contest_list:
            if not entity.get('dialogue_a') or not entity.get('dialogue_b'):
                raise ValueError(f"Invalid contest_list: dialogue_a or dialogue_b is empty in entity {entity}")
        
        self.contest_list=sorted(contest_list, key=lambda x:x['order'])

        for entity in contest_list:
            scores=llm_judge.score(entity['dialogue_a'], entity['dialogue_b'])
            try:
                scores=[eval(choice.message.content) for completion in scores for choice in completion.choices]
            except Exception as e:
                print('reply不符合规范，跳过~')
            score_a, score_b=sum(scores[0]), sum(scores[1])
            win = 1 if score_a > score_b else 0 if score_a < score_b else 0.5
            entity['scores']=scores
            entity['win_status']=win
            
            self._update_ratings(entity['model_a'], entity['model_b'], entity['win_status'])
            self.scores_log.append(copy.deepcopy(self.participants)) #避免list中的元素全部指向同一个变量