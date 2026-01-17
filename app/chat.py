# chat.py - Optimized for your 6GB RTX 4050

import uuid
import torch
import time  # for optional timing
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from app.db import save_psych_profile,get_psych_profile,save_conversation
# ────────────────────────────────────────────────
# CONFIG (your working paths)
# ────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "D:/Coding Workspaces/GSD-A Habit Negotiator AI/models/habit_nudge_qwen2.5_7b_finetuned"
USER_ID = 1  # hardcoded single-user for now
SESSION_ID = str(uuid.uuid4())  # unique per run

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="cuda:0",              # force GPU
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_PATH, is_trainable=False)
model.eval()

print("✅ HabitNudge loaded (7B LoRA, 4-bit, RTX 4050)")

# ────────────────────────────────────────────────
# STRONG SYSTEM PROMPT (your prompts/habit_nudge.py content)
# ────────────────────────────────────────────────
from prompts.habit_nudge import SYSTEM_PROMPT  # import your file

# Simple rolling memory (last 3 full exchanges = 6 messages)
history = []

print("First time? Fill quick psych profile (answer 1-5 or text)")
questions = [
    "How much do you like humor/sarcasm in motivation? (1 low - 5 high): ",
    "Do you respond better to gentle encouragement or tough dares? (gentle / tough): ",
    "How resistant are you usually? (low / medium / high): "
]

profile = {}
for q in questions:
    ans = input(q).strip()
    profile[q.split()[0]] = ans 
    
save_psych_profile(1,profile)
print("Profile saved! I'll use it to adapt my style 😏")
# ────────────────────────────────────────────────
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Bye! Keep those micro-wins coming 💪")
        break

    if not user_input:
        continue

    start_time = time.time()
    
    profile = get_psych_profile(USER_ID)
    if profile:
        print(f"Loaded profile: {profile}")
        style_add = f"User psych: humor {profile.get('How', 'medium')}, prefers {profile.get('Do', 'tough')}, resistance {profile.get('How', 'medium')}. Adapt tone accordingly."
        SYSTEM_PROMPT += "\n" + style_add
    else:
        print("No profile found. Quick psych questions:")
    # ... your questions code here ...
        questions = [
    "How much do you like humor/sarcasm in motivation? (1 low - 5 high): ",
    "Do you respond better to gentle encouragement or tough dares? (gentle / tough): ",
    "How resistant are you usually? (low / medium / high): "
]

        profile = {}
        for q in questions:
            ans = input(q).strip()
            profile[q.split()[0]] = ans 

        save_psych_profile(USER_ID, profile)
        print("Profile saved! I'll use it to adapt my style 😏")
    

# ────────────────────────────────────────────────
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:  # last 3 exchanges
        messages.append(h)
    messages.append({"role": "user", "content": user_input})
    
    last_goal = None
    last_micro = None
    last_barriers = []
# Inside the loop, after printing response
    save_conversation(
    user_id=USER_ID,
    session_id=SESSION_ID,
    messages=[{"role": m["role"], "content": m["content"]} for m in messages],  # current full context
    goal="run everyday 5km" if "run" in user_input.lower() else last_goal,  # extract or keep last
    micro=last_micro,  # parse from response if possible
    barriers=last_barriers
)
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda:0")

    attention_mask = torch.ones_like(input_ids).to("cuda:0")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=70,                # tight limit
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.15,          # stronger anti-repetition
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True                    # re-enabled – big speedup
        )

    response = tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"\nHabitNudge ({time.time() - start_time:.1f}s):", response, "\n")

    # Update history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    # Optional: trim history if too long
    if len(history) > 12:
        history = history[-12:]