# coding: utf-8
"""Generate an answer from a reasoning structure and optional context."""
from prompts import generation_answer, generation_answer_rag


def solve_problem(llm, problem, reasoning_structure, gold_context=None, dataset=None):
    if gold_context:
        prompt = generation_answer_rag.replace("{context}", gold_context)
    else:
        prompt = generation_answer
    prompt = prompt.replace("{problem}", problem)
    prompt = prompt.replace("{reasoning_structure}", reasoning_structure.strip())

    if dataset == "StrategyQA":
        prompt = prompt.strip() + "\nAnswer with only one word: True or False."
    elif dataset == "BBH_web_of_lies":
        prompt = prompt.strip() + "\nAnswer with only one word: Yes or No."
    elif dataset == "BBH_formal_fallacies":
        prompt = prompt.strip() + "\nAnswer with only one word: Valid or Invalid."

    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        try:
            response, _ = llm.generate_response(prompt)
            return response.strip()
        except Exception:
            retry_count += 1
            print(f"retry {retry_count}")    
    else:
        return None
    