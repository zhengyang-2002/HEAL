import Prompt
from openai import OpenAI


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


deepseek=Judge("deepseek-chat","https://api.deepseek.com/v1","sk-d499467c635a420c93adeffbf28e6939")

sample_a='''
来访者：我确实没有一个合理的复习计划，每天都觉得时间不够用，也不知道该从何开始。你能给我一些建议吗？
咨询师：首先，你要明确自己的目标，既然你想考本科，那就要为此做好准备。然后，你需要制定一个合理的复习计划，根据自己的情况来安排每天的学习时间和内容。这样可以帮助你更好地掌控整个复习过程，减少焦虑感。
来访者：我明白了，但具体该怎么制定计划呢？
咨询师：当然可以！你可以从高一开始，试试题海战术。每天多做一些题目，这样能够提高你的学习效率。同时，对于英语这门科目，多听多背是很重要的，数理化方面，可以做一些经典的题目，特别是那些类型经常考到的题目，多次反复做题。
来访者：听起来很有帮助，我会试试的。
咨询师：很好，试试看吧。如果遇到困难或需要调整计划，随时可以来找我讨论。希望这些建议能帮助你更好地准备考试！
'''

sample_b='''
来访者：你好
咨询师：你好，很高兴能够帮助到你。你可以告诉我更多关于你的情况吗？
来访者：我最近睡眠很差怎么办？
咨询师：最近睡眠质量不好，你有没有尝试过一些方法来改善睡眠呢？
来访者：我可能最近晚上睡觉前总喜欢玩手机，越玩人就越兴奋
咨询师：晚上玩手机确实会刺激大脑，导致入睡困难。你可以尝试在睡觉前设定一个手机使用时间，逐渐减少玩手机的时间，以帮助放松入睡。
来访者：我应该设置一个晚上9点的闹钟，强制我不玩手机，让我的大脑开始放松。
咨询师：是的，设置一个晚上的闹钟是一个很好的方法。你可以尝试在9点之前放下手机，让自己进入放松的状态，然后逐渐入睡。
来访者：我 尝 试了一下，看看能否改善？
咨询师：很好，试试看吧。另外，你还可以尝试一些放松的技巧，比如深呼吸、冥想或者听一些轻柔的音乐来帮助你放松身心，帮助入睡。
来访者：非常感谢你的建议，我会去试试的
咨询师：不客气，希望这些方法能对你有所帮助。如果还有其他问题，随时都可以来找我倾诉。
'''

%%time
result=deepseek.score(sample_a,sample_b)
print(result)

class Participant:
    def __init__(self, name, inital_rating=1000):
        self.name=name
        self.rating=inital_rating
        
    def update_rating(self, rating_change):
        self.rating+=rating_change

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

# 示例用法
model1 = Participant("Model A")
model2 = Participant("Model B")

# 示例用法
model1 = LanguageModel("Model A")
model2 = LanguageModel("Model B")

elo_system = EloRatingSystem()

# 假设Model A赢了
elo_system.match(model1, model2, 1)

print(f"{model1.name} 的新评分: {model1.rating}")
print(f"{model2.name} 的新评分: {model2.rating}")