def load_messages(model_name, prompt):
    templates = {
        # Models run locally
        "gpt-oss-20b": [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": prompt}
        ],
        "qwen3-14b": [
            {"role": "user", "content": prompt}
        ],
        "llama-8b": [
            {"role": "user", "content": prompt}
        ],
    }
    return templates.get(model_name)

