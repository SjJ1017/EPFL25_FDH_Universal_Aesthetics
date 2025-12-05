
import os
import json
import openai
from tqdm import tqdm

def read_csv_to_dicts(csv_path):
    """Read a CSV file and return a list of dictionaries, one per row."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_dict(orient='records')



def estimate_gpt4o_cost_with_prompt(poems,
                                    avg_output_tokens=150,
                                    price_in=0.005,
                                    price_out=0.015):
    """

    参数：
        poems: list[str]  诗歌文本列表
        avg_output_tokens: int 模型平均输出 token 数（默认 150）
        price_in: float   输入 token 单价（$/1k tokens）
        price_out: float  输出 token 单价（$/1k tokens）

    返回：
        total_tokens_in, total_tokens_out, total_cost
    """


    prompt_template = """You are an objective literary evaluator.
Task:
1) Evaluate the beauty of the following poem and give it an integer score from 1 to 5 (1 = not beautiful, 5 = extremely beautiful).
2) First output a concise, non-sensitive rationale (a clear summary explaining why you gave this score). This should be at most ~200 words, avoid step-by-step internal chain-of-thought.
3) Then output the score.
4) Finally return a JSON object EXACTLY in this format (no extra commentary):
{{ "thinking": "<the concise rationale as a string>", "score": <int 1-5> }}

Poem to evaluate:
\"\"\"{poem_text}\"\"\"

Scoring criteria (you MUST apply these; briefly mention which criteria influenced the score in the rationale):
- Imagery & Sensory Detail (weight 30%): quality and vividness of images, sensory language.
- Emotional Impact (weight 25%): emotional resonance, ability to move reader.
- Language & Diction (weight 15%): word choice, originality, metaphors, semantic richness.
- Structure & Rhythm (weight 15%): line breaks, meter/flow, internal cohesion.
- Originality & Depth (weight 15%): fresh perspective or depth of thought.

Scoring method: evaluate each criterion on 0-10, compute weighted sum, map to 1-5:
  total_score_0_10 = weighted average (0-10)
  final_score = round(total_score_0_10 / 2)

Important instructions for your output:
- Do NOT reveal internal chain-of-thought.
- Output MUST be valid JSON as specified in step 4.
Now perform the evaluation.
"""

    def chars_to_tokens(chars: int):
        return chars / 4

    total_tokens_in = 0
    total_tokens_out = 0

    for poem in poems:
        prompt_filled = prompt_template.format(poem_text=poem)
        prompt_tokens = chars_to_tokens(len(prompt_filled))
        total_tokens_in += prompt_tokens
        total_tokens_out += avg_output_tokens

    cost_in = total_tokens_in / 1000 * price_in
    cost_out = total_tokens_out / 1000 * price_out
    total_cost = cost_in + cost_out

    return round(total_tokens_in), round(total_tokens_out), round(total_cost, 4)


def evaluate_poem(poem_text: str) -> dict:

    prompt = f"""
    You are an objective literary evaluator. 
    Task:
    1) Evaluate the beauty of the following poem and give it an integer score from 1 to 5 (1 = not beautiful, 5 = extremely beautiful).
    2) First output a concise, non-sensitive rationale (a clear summary explaining why you gave this score). This should be at most ~200 words, avoid step-by-step internal chain-of-thought.
    3) Then output the score.
    4) Finally return a JSON object EXACTLY in this format (no extra commentary):
    {{ "thinking": "<the concise rationale as a string>", "score": <int 1-5> }}

    Poem to evaluate:
    \"\"\"{poem_text}\"\"\"

    Scoring criteria (you MUST apply these; briefly mention which criteria influenced the score in the rationale):
    - Imagery & Sensory Detail (weight 30%): quality and vividness of images, sensory language.
    - Emotional Impact (weight 25%): emotional resonance, ability to move reader.
    - Language & Diction (weight 15%): word choice, originality, metaphors, semantic richness.
    - Structure & Rhythm (weight 15%): line breaks, meter/flow, internal cohesion.
    - Originality & Depth (weight 15%): fresh perspective or depth of thought.

    Scoring method: evaluate each criterion on 0-10, compute weighted sum, map to 1-5:
    total_score_0_10 = weighted average (0-10)
    final_score = round(total_score_0_10 / 2)  # maps 0-10 to 0-5, round to nearest int, but clamp to 1-5

    Important instructions for your output:
    - Do NOT reveal internal chain-of-thought. Only provide a concise rationale (summary) explaining which criteria mattered and how.
    - Output MUST be valid JSON as specified in step 4. 'thinking' must be a string, 'score' an integer.
    - Example of allowed rationale: "Strong imagery and emotional resonance; language sometimes cliché; good rhythm; overall score 4."

    Now perform the evaluation.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an objective, concise literary critic."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=700
    )

    text = response["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}$", text)
        if m:
            parsed = json.loads(m.group(0))
        else:
            return {"thinking": "Error parsing model output.", "score": 0}

    thinking = parsed.get("thinking", "")
    score = int(parsed.get("score", 0))

    result = {"thinking": thinking, "score": score}
    return result

def load_multiline_jsonl(path):
    data = []
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            buffer += line
            try:
                obj = json.loads(buffer)
                data.append(obj)
                buffer = "" 
            except json.JSONDecodeError:
                continue
    if buffer.strip():
        raise ValueError("Incomplete JSON object at end of file")
    return data


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    dataset = '/poems/cleaned_poems.csv'
    poems_data = read_csv_to_dicts(dataset)

    for poem_entry in tqdm(poems_data):
        filename = poem_entry.get('filename', '')
        poem_text = poem_entry['content']
        eval_result = evaluate_poem(poem_text)
        poem_entry['evaluation'] = eval_result
        with open('poems/rated_poems_results_sampled.jsonl', 'a') as f:
            f.write(json.dumps(poem_entry, indent=4) + '\n')