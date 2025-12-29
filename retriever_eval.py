import re
import numpy as np
from typing import List, Dict

# ---------------------------
# Helpers
# ---------------------------
CEFR_LEXICON = {
    "A1": ["먹다", "끓이다", "굽다", "고기", "물","밥","닭","돼지","소고기","야채","채소","국","찌개","자르다","넣다","빼다","많이","적게"],
    "A2": ["볶다", "재료", "요리", "불", "기름","양념","간장","고추장","된장","마늘","파","중불","약불","센불","볶음","튀김","국물","조리","시간"],
    "B1": ["어패류", "돈육", "무육","숙성","염지","발표","저온조리","데치다","블랜칭","팬프라잉","시어링","마리네이드","육수","해감","비린내","식감","풍미","조화"],
    "B2": ["겉바속촉", "단짠단짠", "비주얼","칼칼하다","담백하다","감칠맛","불향","고급스러운","근사하다","플레이팅","한상차림","집들이용","손님접대용"],
}

def parse_ingredients(text: str) -> List[str]:
    items = re.split(r"[,/\\|\n]+", text)
    return [i.strip() for i in items if i.strip()]

def ingredient_match_ratio(doc_text: str, user_ings: List[str]) -> float:
    if not user_ings:
        return 0.0
    hit = sum(1 for ing in user_ings if ing in doc_text)
    return hit / len(user_ings)

def difficulty_score(level: str) -> float:
    if not level:
        return 0.0
    if level in ["초급", "아무나", "쉬움", "Easy"]:
        return 1.0
    if level in ["중급"]:
        return 0.5
    return 0.0

def popularity_score(views: int, threshold: int = 5000) -> float:
    if views <= 0:
        return 0.0
    return min(1.0, views / threshold)

def cefr_score(text: str) -> float:
    if not text:
        return 1.0

    total = 0
    score = 0.0

    for level, words in CEFR_LEXICON.items():
        for w in words:
            if w in text:
                total += 1
                if level == "A1":
                    score += 1.0
                elif level == "A2":
                    score += 0.8
                elif level == "B1":
                    score += 0.4
                else:
                    score += 0.1

    if total == 0:
        return 1.0
    return round(score / total, 3)

# ---------------------------
# Main Evaluation
# ---------------------------
def evaluate_retriever(
    retriever,
    query: str,
    ingredients: str,
    top_k: int = 5,
    verbose: bool = True
) -> Dict[str, float]:

    user_ings = parse_ingredients(ingredients)
    docs = retriever.invoke(query)[:top_k]

    ips_scores = []
    dps_scores = []
    pps_scores = []

    for idx, d in enumerate(docs, 1):
        text = d.page_content or ""
        md = d.metadata or {}

        ips = ingredient_match_ratio(text, user_ings)
        dps = difficulty_score(md.get("level", ""))
        pps = popularity_score(int(md.get("views", 0) or 0))

        ips_scores.append(ips)
        dps_scores.append(dps)
        pps_scores.append(pps)

        if verbose:
            print(f"\n--- Doc {idx} ---")
            print(f"Menu      : {md.get('menu')}")
            print(f"Level     : {md.get('level')}")
            print(f"Views     : {md.get('views')}")
            print(f"IPS       : {ips:.2f}")
            print(f"DPS       : {dps:.2f}")
            print(f"PPS       : {pps:.2f}")

    return {
        "IPS@K": round(float(np.mean(ips_scores)), 3),
        "DPS@K": round(float(np.mean(dps_scores)), 3),
        "PPS@K": round(float(np.mean(pps_scores)), 3),
    }
