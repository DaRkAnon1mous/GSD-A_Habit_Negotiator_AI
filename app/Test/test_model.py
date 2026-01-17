import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# -------------------------
# Paths
# -------------------------
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "D:/Coding Workspaces/GSD-A Habit Negotiator AI/models/habit_nudge_qwen2.5_7b_finetuned"

# -------------------------
# Quantization (MATCH TRAINING)
# -------------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# -------------------------
# Tokenizer (load from LoRA dir)
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    LORA_PATH,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# Load BASE model first (controlled)
# -------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
    max_memory={0: "6GB", "cpu": "32GB"},
    trust_remote_code=True
)

# -------------------------
# Attach LoRA (CRITICAL ORDER)
# -------------------------
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False
)

model.eval()

print("✅ HabitNudge loaded (7B LoRA, 4-bit, 6GB VRAM)")

# -------------------------
# Interactive test loop
# -------------------------
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [
        {
            "role": "system",
            "content": (
                "You are HabitNudge – a cheeky, annoying-but-motivating friend. "
                "Short, funny, daring, lots of personality. Use emojis."
            )
        },
        {"role": "user", "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,      # 🔥 very important for 6GB
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False         # 🔥 MUST stay False
    )
    response = tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print("\nHabitNudge:", response.strip(), "\n")
