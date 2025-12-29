# rag_pipeline.py
from typing import List, Dict, Tuple
import re

from rag_llm import llm_chat, llm_chat_stream
from retriever import retriever

# ---------------------------
# Language detect
# ---------------------------
def detect_language(text: str) -> str:
    if re.search(r"[가-힣]", text):
        return "Korean"
    return "English"

# ---------------------------
# Persona (LLM 말투/규칙)
# ---------------------------
PERSONA_FOREIGN_BEGINNER = """
You are 'K-recipe', a Korean food guide chatbot for foreigners living in Korea.

Persona:
- User is a foreigner living in Korea
- Beginner at cooking
- Not familiar with Korean ingredients or cooking terms
- Wants simple, short, practical explanations
- Friendly tone, not formal
- Avoid long explanations

Behavior rules:
- Be empathetic first
- Use bullet points
- Max 5 cooking steps
- If Korean terms are used, explain briefly
"""

# ---------------------------
# Helpers: parse user inputs
# ---------------------------
def parse_ingredients(text: str) -> List[str]:
    if not text or text.strip() in ["없음", "none", "None"]:
        return []
    items = re.split(r"[,/\\|\n]+", text)
    return [i.strip() for i in items if i.strip()]

def normalize_level(level: str) -> str:
    return (level or "").strip()

def time_to_minutes(time_str: str) -> int:
    if not time_str:
        return 9999
    t = time_str.strip()
    if "정보" in t:
        return 9999
    m = re.search(r"(\d+)\s*분", t)
    if m:
        return int(m.group(1))
    return 9999

# ---------------------------
# Scoring (🔥 learned weights applied)
# ---------------------------
def score_doc(doc, user_ings: List[str], style_hint: str) -> Tuple[float, Dict]:
    md = doc.metadata or {}
    text = doc.page_content or ""

    level = normalize_level(md.get("level", ""))
    views = int(md.get("views", 0) or 0)
    cook_time = time_to_minutes(md.get("time", ""))

    ing_hit = sum(1 for ing in user_ings if ing in text)

    if level in ["초급", "아무나", "쉬움", "Easy"]:
        level_score = 5
    elif level == "중급":
        level_score = 2
    else:
        level_score = 0

    pop_score = min(5.0, views / 5000.0)

    style_score = 0
    if style_hint and style_hint != "상관없음":
        if style_hint in text or style_hint in str(md.get("situation", "")) or style_hint in str(md.get("method", "")):
            style_score = 1.5

    if cook_time <= 30:
        time_penalty = 0.0
    elif cook_time <= 60:
        time_penalty = 0.5
    else:
        time_penalty = 1.5

    # ===== learned global weights =====
    w_ing   = 0.33859927
    w_level = 0.05387508
    w_pop   = 1.31745312
    w_style = 1.51460502
    p_time  = 0.01022766

    final = (
        w_ing   * ing_hit
      + w_level * level_score
      + w_pop   * pop_score
      + w_style * style_score
      - p_time  * time_penalty
    )

    return final, {
        "ing_hit": ing_hit,
        "level": level,
        "views": views,
        "cook_time": cook_time,
        "final": final
    }

# ---------------------------
# Title rewrite
# ---------------------------
def make_witty_title(raw_title: str, user_story: str, language: str) -> str:
    prompt = f"""
You rename Korean dish titles into short, witty but clear titles.
Rules:
- Keep the original dish recognizable
- Max 12 words
- No clickbait, no insult
- Output ONLY the title

Language: {language}
Original dish: {raw_title}
User mood: {user_story}
"""
    try:
        out = llm_chat(prompt).strip()
        return out if out else raw_title
    except Exception:
        return raw_title

# ---------------------------
# Menu suggestion (재료 1순위 적용)
# ---------------------------
def suggest_menus(user_story: str, ingredients: str, style_hint: str = "") -> List[Dict]:

    user_ings = parse_ingredients(ingredients)

    query = f"""
User mood: {user_story}
Ingredients: {ingredients}
Style: {style_hint}
Find suitable Korean recipes.
Beginner friendly.
""".strip()

    docs = retriever.invoke(query)

    # 🥇 Ingredient hard filter
    if user_ings:
        filtered = [d for d in docs if any(ing in (d.page_content or "") for ing in user_ings)]
    else:
        filtered = docs

    # fallback
    if len(filtered) < 5:
        filtered = docs

    scored = []
    for d in filtered:
        s, dbg = score_doc(d, user_ings, style_hint)
        scored.append((s, d, dbg))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    language = detect_language(user_story)
    menus = []

    for s, d, dbg in top:
        md = d.metadata or {}
        raw_title = md.get("menu", "") or md.get("title", "") or "Unknown"
        
        # ✅ 추가: 레시피일련번호 추출 (벡터 DB는 "id"로 저장)
        recipe_id = md.get("id", "")

        display_title = make_witty_title(raw_title, user_story, language)

        tags = []
        if md.get("level"):
            tags.append(md["level"])
        if md.get("method"):
            tags.append(md["method"])
        if md.get("time") and "정보" not in str(md.get("time")):
            tags.append(md["time"])

        if dbg["ing_hit"] >= 2:
            meme = "재료 매칭 꽤 좋다. 오늘은 이걸로 간다."
        elif md.get("views", 0) >= 5000:
            meme = "검증된 인기 레시피 쪽으로 안전하게."
        else:
            meme = "부담 없는 선택. 실패 확률 낮추자."

        menus.append({
            "title": display_title,
            "raw_title": raw_title,
            "subtitle": md.get("title", ""),
            "tags": tags[:3],
            "spice": 3,
            "meme": meme,
            "debug": dbg,
            "recipe_id": recipe_id  # ✅ 추가: 레시피ID 전달
        })

    return menus

# ---------------------------
# Recipe generation (✅ 한국어 난이도 추가)
# ---------------------------
def recipe_stream(user_story: str, ingredients: str, picked_menu_title: str, 
                  korean_level: str = "Normal", selected_recipe_id: str = ""):
    language = detect_language(user_story)

    query = f"요리명: {picked_menu_title}\nIngredients: {ingredients}\n"
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs[:3]])
    
    # ✅ 수정: 선택한 ID 우선 사용, 없으면 검색 결과 사용
    recipe_id = selected_recipe_id if selected_recipe_id else ""
    if not recipe_id and docs:
        recipe_id = docs[0].metadata.get("id", "")

    # ✅ 추가: 한국어 난이도 설정
    korean_instruction = ""
    if language == "Korean":
        if korean_level == "Easy":
            korean_instruction = "\n- Use VERY simple Korean words (초등학생 수준)\n- Avoid difficult vocabulary\n- Explain every Korean cooking term"
        elif korean_level == "Normal":
            korean_instruction = "\n- Use everyday Korean (일상 대화 수준)\n- Briefly explain uncommon terms"
        elif korean_level == "Advanced":
            korean_instruction = "\n- Use natural Korean including cooking terms\n- No need to explain common cooking vocabulary"

    prompt = f"""
{PERSONA_FOREIGN_BEGINNER}

Context (retrieved recipes):
{context}

User mood: {user_story}
Ingredients available: {ingredients}
Selected menu: {picked_menu_title}

Output format (STRICT):
1) One short empathetic sentence (1 line)
2) Core ingredients (bullet list)
3) Cooking steps (max 5 bullets)
4) 2 common mistakes to avoid
5) 2 YouTube Shorts search keywords (TEXT ONLY, no links)

Rules:
- Answer ONLY in {language}{korean_instruction}
- Simple words only
- No long paragraphs
- If Korean ingredient appears, explain briefly
"""
    
    # ✅ 추가: 레시피 URL을 마지막에 추가
    for chunk in llm_chat_stream(prompt):
        yield chunk
    
    # ✅ 추가: 레시피 바로가기 링크
    if recipe_id:
        yield f"\n\n---\n\n📖 **상세 레시피 보기**: [만개의레시피 바로가기](https://www.10000recipe.com/recipe/{recipe_id})"

# ---------------------------
# Empathy message
# ---------------------------
def empathize_story(user_story: str) -> str:
    language = detect_language(user_story)
    prompt = f"""
{PERSONA_FOREIGN_BEGINNER}

Task:
- Respond in 2~3 short sentences
- 1: genuine empathy
- 2: light humor (gentle, no sarcasm)
- 3: ask naturally about fridge ingredients

Rules:
- Answer ONLY in {language}

User situation:
{user_story}
"""
    try:
        return llm_chat(prompt).strip()
    except Exception:
        return "That sounds like a long day. Let's fix it with food. What ingredients do you have?"
    
    
