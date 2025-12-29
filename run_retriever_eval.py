import pandas as pd
from retriever_eval import evaluate_retriever
from eval_scenarios import SCENARIOS
from retriever import retriever

results = []

for sc in SCENARIOS:
    print(f"\n==============================")
    print(f"Evaluating: {sc['name']}")
    print(f"Query     : {sc['query']}")
    print(f"Ingredients: {sc['ingredients']}")

    scores = evaluate_retriever(
        retriever=retriever,
        query=sc["query"],
        ingredients=sc["ingredients"],
        top_k=5,
        verbose=False
    )

    results.append({
        "scenario": sc["name"],
        **scores
    })

df_eval = pd.DataFrame(results)

from rag_pipeline import recipe_stream
from retriever_eval import cefr_score
gen_scores = []

for sc in SCENARIOS:
    output = ""
    for chunk in recipe_stream(
        user_story=sc["query"],
        ingredients=sc["ingredients"],
        picked_menu_title="임의 메뉴",
        lang_level="BEGINNER"
    ):
        output += chunk

    gen_scores.append({
        "scenario": sc["name"],
        "CEFR_score": cefr_score(output)
    })

df_gen = pd.DataFrame(gen_scores)

df_final = df_eval.merge(df_gen, on="scenario")

print("\n===== FINAL EVALUATION RESULT =====\n")
print(df_final)
