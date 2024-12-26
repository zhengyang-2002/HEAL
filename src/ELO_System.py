from openai import OpenAI
class LanguageModel:
    def __init__(self, name, rating=1000):
        self.name = name
        self.rating = rating

    def update_rating(self, score_change):
        self.rating += score_change




class EloRatingSystem:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor

    def expected_score(self, model_a, model_b):
        return 1 / (1 + 10 ** ((model_b.rating - model_a.rating) / 400))

    def update_ratings(self, model_a, model_b, result):
        expected_a = self.expected_score(model_a, model_b)
        expected_b = self.expected_score(model_b, model_a)

        score_change_a = self.k_factor * (result - expected_a)
        score_change_b = self.k_factor * ((1 - result) - expected_b)

        model_a.update_rating(score_change_a)
        model_b.update_rating(score_change_b)

    def match(self, model_a, model_b, result):
        """
        result: 1 if model_a wins, 0.5 for a draw, 0 if model_b wins
        """
        self.update_ratings(model_a, model_b, result)

# 示例用法
model1 = LanguageModel("Model A")
model2 = LanguageModel("Model B")

elo_system = EloRatingSystem()

# 假设Model A赢了
elo_system.match(model1, model2, 1)

print(f"{model1.name} 的新评分: {model1.rating}")
print(f"{model2.name} 的新评分: {model2.rating}")
