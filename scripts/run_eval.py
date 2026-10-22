import json
import re
import unicodedata
import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============== 1. 路径配置 ==============
QA_JSON = r"F:\BS\docs_lm_ko\data\eval\ko_doc_qa.json"
PAGE_DIR = r"F:\BS\docs_lm_ko\outputs\pages"


# ============== 2. 评估指标函数（已整合，无需外部依赖） ==============
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'(입니다|합니다|라고 합니다|인 것 같습니다|으로 나타났습니다)(\.|!|\?|\.*)?$', '', text)
    return text.strip()


def normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'[^가-힣a-zA-Z0-9]', '', s)
    return s.lower()


def exact_match(pred: str, golds) -> float:
    p = normalize(clean_text(pred))
    for g in golds:
        if p == normalize(g):
            return 1.0
    return 0.0


def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return max(m, n)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[m][n]


def anls(pred: str, golds, theta=0.5) -> float:
    p = normalize(clean_text(pred))
    scores = []
    for g in golds:
        g_norm = normalize(g)
        l = max(len(p), len(g_norm))
        if l == 0:
            scores.append(1.0)
            continue
        dist = levenshtein_distance(p, g_norm)
        s = 1.0 - (dist / l)
        scores.append(s if s >= theta else 0.0)
    return max(scores) if scores else 0.0


# ============== 3. 加载模型与数据 ==============
print("Loading Qwen2-VL Model...")
model_id = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

with open(QA_JSON, encoding="utf-8") as f:
    qa_list = json.load(f)

# ============== 4. 逐条推理与计算 ==============
preds_golds = []
total_entropy = 0.0
entropy_count = 0

for item in qa_list:
    img_path = f"{PAGE_DIR}/{item['page']}.png"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": f"질문에 한국어로 간결하게 답하세요: {item['question']}"}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            return_dict_in_generate=True,
            output_scores=True
        )

    gen_tokens = out.sequences[0][inputs.input_ids.shape[1]:]
    pred = processor.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"Q{item['id']}: {pred}")

    # 计算熵
    if out.scores is not None and len(out.scores) > 0:
        avg_entropy = 0.0
        for score in out.scores:
            probs = F.softmax(score[0], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            avg_entropy += entropy.item()
        avg_entropy /= len(out.scores)
        total_entropy += avg_entropy
        entropy_count += 1

    preds_golds.append((pred, item["answer"]))

# ============== 5. 汇总并打印结果 ==============
ems, anls_scores = [], []
keyword_match_count = 0

for pred, golds in preds_golds:
    ems.append(exact_match(pred, golds))
    anls_scores.append(anls(pred, golds))
    for g in golds:
        if g in pred:
            keyword_match_count += 1
            break

if not ems:
    final_em, final_anls = 0.0, 0.0
else:
    final_em = round(sum(ems) / len(ems), 4)
    final_anls = round(sum(anls_scores) / len(anls_scores), 4)

km = round(keyword_match_count / len(preds_golds), 4) if preds_golds else 0.0
avg_entropy = round(total_entropy / entropy_count, 4) if entropy_count > 0 else 0.0

print("\n=== 결과 ===")
print(json.dumps({"EM": final_em, "ANLS": final_anls}, ensure_ascii=False, indent=2))
print(f"'Keyword Match (KM)': {km}")
print(f"'Avg Entropy (불확실성)': {avg_entropy}")