import random
import numpy as np
from rag_pipeline import suggest_menus
from retriever import retriever

N_TEST = 1000
TOP_K = 5

# ------------------------------
# 자동 재료 샘플러 (DB에서 추출)
# ------------------------------
def sample_ingredients_from_db():
    docs = retriever.invoke("재료")
    texts = " ".join(d.page_content for d in docs)

    candidates = []
    for token in texts.split():
        if len(token) >= 2 and token.isalpha():
            candidates.append(token)

    if not candidates:
        return ["김치"]

    return random.sample(candidates, random.randint(1, 3))

STYLES = ["상관없음", "초간단", "든든한 한 끼", "혼술 안주", "칼칼/매콤"]

# ------------------------------
# Metric 계산
# ------------------------------
def compute_metrics(menus, user_ings):
    if not menus:
        return 0, 0, 0

    # Ingredient Priority Score
    ing_hits = []
    for m in menus:
        hit = 1 if any(ing in str(m) for ing in user_ings) else 0
        ing_hits.append(hit)
    IPS = sum(ing_hits) / len(menus)

    # Difficulty Priority Score
    easy_cnt = sum(1 for m in menus if any(t in ["초급", "아무나", "쉬움", "Easy"] for t in m["tags"]))
    DPS = easy_cnt / len(menus)

    # Popularity Priority Score
    views = [m["debug"].get("views", 0) for m in menus]
    PPS = np.mean(views)

    return IPS, DPS, PPS

# ------------------------------
# 실험 루프
# ------------------------------
ips, dps, pps = [], [], []

for _ in range(N_TEST):
    user_ings = sample_ingredients_from_db()
    style = random.choice(STYLES)

    menus = suggest_menus(
        user_story="오늘 집밥 먹고 싶다",
        ingredients=",".join(user_ings),
        style_hint=style
    )

    I, D, P = compute_metrics(menus, user_ings)

    ips.append(I)
    dps.append(D)
    pps.append(P)

# ------------------------------
# 결과 출력
# ------------------------------
print("\n===== AUTO PRIORITY EVALUATION =====")
print(f"Ingredient Priority Score: {np.mean(ips):.3f} ± {np.std(ips):.3f}")
print(f"Difficulty Priority Score: {np.mean(dps):.3f} ± {np.std(dps):.3f}")
print(f"Popularity Priority Score: {np.mean(pps):.1f} ± {np.std(pps):.1f}")
