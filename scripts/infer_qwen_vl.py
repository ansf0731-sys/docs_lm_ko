import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen2-VL-2B-Instruct"
img = r"C:\Users\AN-SF\docs_lm_ko\outputs\pages\page1.png"

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img, "min_pixels": 256*28*28, "max_pixels": 1024*28*28},
        {{"type": "text", "text": "이 페이지에 표가 있으면 표의 열 이름과 각 행의 값을 한국어로 정리해 주세요. 표가 없으면 없다고 답하세요."}}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
print("keys:", inputs.keys())
print("pixel_values:", None if inputs.get("pixel_values") is None else inputs["pixel_values"].shape)
print("image_grid_thw:", inputs.get("image_grid_thw"))
out = model.generate(**inputs, max_new_tokens=256)
gen = out[0][inputs.input_ids.shape[1]:]
print(processor.decode(gen, skip_special_tokens=True))