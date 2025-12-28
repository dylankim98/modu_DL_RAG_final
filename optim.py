import random
import numpy as np
from scipy.optimize import differential_evolution
from retriever import retriever


# ===========================
# Auto Ranker
# ===========================

class AutoRanker:
    def __init__(self):
        # [w_ing, w_level, w_pop, w_style, p_time]
        self.params = np.array([2.0, 1.0, 0.5, 1.0, 1.0])

    def _level_score(self, level):
        if level in ["초급", "아무나", "쉬움", "Easy"]:
            return 1
        if level == "중급":
            return 0.5
        return 0

    def _parse_time(self, t):
        if "30" in str(t):
            return 0
        if "60" in str(t):
            return 1
        return 2

    def score(self, doc, user_ings, style_hint):
        md = doc.metadata or {}
        text = doc.page_content or ""

        ing_hit = sum(1 for ing in user_ings if ing in text)
        level_score = self._level_score(md.get("level", ""))
        pop_score = np.log1p(int(md.get("views", 0)))

        style_score = 1 if style_hint and (
            style_hint in text
            or style_hint in str(md.get("method", ""))
            or style_hint in str(md.get("situation", ""))
        ) else 0

        time_pen = self._parse_time(md.get("time", ""))

        w_ing, w_level, w_pop, w_style, p_time = self.params

        return (
            w_ing * ing_hit
            + w_level * level_score
            + w_pop * pop_score
            + w_style * style_score
            - p_time * time_pen
        )

    # ===========================
    # Global Objective
    # ===========================

    def _objective(self, params, scenarios):
        self.params = params

        total_score = 0

        for docs, user_ings, style_hint in scenarios:
            ranked = sorted(docs, key=lambda d: self.score(d, user_ings, style_hint), reverse=True)
            top = ranked[:5]

            views = np.mean([int(d.metadata.get("views", 0)) for d in top])
            ing_match = np.mean([sum(1 for ing in user_ings if ing in (d.page_content or "")) for d in top])
            style_match = np.mean([
                1 if style_hint and (
                    style_hint in (d.page_content or "")
                    or style_hint in str(d.metadata.get("situation", ""))
                    or style_hint in str(d.metadata.get("method", ""))
                ) else 0 for d in top
            ])
            level_match = np.mean([
                1 if d.metadata.get("level") in ["초급", "아무나", "쉬움", "Easy"] else 0
                for d in top
            ])

            total_score += views + 2000 * ing_match + 1000 * style_match + 500 * level_match

        return -total_score


# ===========================
# Scenario Generator
# ===========================

STYLES = ["상관없음", "초간단", "든든한 한 끼", "혼술 안주", "칼칼/매콤"]

def generate_scenarios(n=50):
    scenarios = []

    for _ in range(n):
        ingredients = random.sample(["김치", "두부", "돼지고기", "양파", "계란", "감자", "치즈", "고추장", "대파"], random.randint(1, 3))
        style = random.choice(STYLES)

        query = f"""
        Ingredients: {','.join(ingredients)}
        Style: {style}
        Find suitable Korean recipes.
        Beginner friendly.
        """

        docs = retriever.invoke(query)
        scenarios.append((docs, ingredients, style))

    return scenarios


# ===========================
# Train Global Ranker
# ===========================

if __name__ == "__main__":

    print("\n🧠 Generating training scenarios...")
    scenarios = generate_scenarios(60)

    ranker = AutoRanker()

    bounds = [(0, 5), (0, 5), (0, 2), (0, 3), (0, 3)]

    print("🚀 Optimizing global ranking weights...\n")

    result = differential_evolution(
        ranker._objective,
        bounds=bounds,
        args=(scenarios,),
        maxiter=60,
        polish=True
    )

    best_params = result.x
    np.save("ranker_weights.npy", best_params)

    print("\n✅ Training complete.")
    print("🏆 Learned Global Weights:")
    print(f"[w_ing, w_level, w_pop, w_style, p_time] = {best_params}")
