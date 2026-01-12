import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer,SFTConfig

# Config
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
DATA_PATH = "D:/Coding Workspaces/GSD-A Habit Negotiator AI/data/habit_nudge_dataset.jsonl"
OUTPUT_DIR = "D:/Coding Workspaces/GSD-A Habit Negotiator AI/models/habit_nudge_qwen2.5_7b_finetuned"
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

# Quantization (4-bit)
# Goal: Make a 3-billion-parameter model fit in a gaming GPU (6-8 GB instead of 12 GB).
# What it does: Every weight becomes 4 tiny bits instead of 32 big bits while it sits in VRAM.
# Why NF4: It’s a special 4-bit language that keeps the shape of the numbers, so quality loss is tiny.
# Why FP16 compute: Your GPU’s “math brain” (Tensor Cores) likes half-precision; we temporarily blow the 4-bit bricks back to 16-bit only for the multiplication, then shrink them again.
# Double quant: Even the rulers (scale factors) are stored in 8-bit → another ~500 MB saved for free.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, # Shrink every weight to 4 bits, A 7-B model that would need ~14 GB of GPU memory now needs ~3–4 GB.
    bnb_4bit_quant_type="nf4", # Use the ‘Normal-Floating 4-bit’ encoding NF4 is a special 4-bit format designed by the bitsandbytes authors.It stores values in a non-uniform way: more precision near 0, less in the tails → keeps the distribution shape of the original weights, so quality drops less than plain INT4.
    bnb_4bit_compute_dtype=torch.float16, # Do the math in FP16, Weights are 4-bit, but when the GPU multiplies matrices it dequantizes to float16 first.Faster than dequantizing to float32 and still accurate enough for most LLMs.
    bnb_4bit_use_double_quant=True, # Compress the compression constants, 4-bit quantization needs a small scale/zero vector per weight block (typically 64 weights).Double-quant stores those scale vectors themselves in 8-bit, saving another ~0.3–0.5 GB on a 7-B model
    # llm_int8_enable_fp32_cpu_offload=True # If your GPU is too small, it can temporarily shove some weights to CPU RAM during forward passes.
)

# LoRA config 
lora_config = LoraConfig(
    r=16, #each strip is only 32 columns wide (tiny!)
    lora_alpha=32, #we amplify the strip’s signal by 2× so the model still “feels” it.
    target_modules=["q_proj","v_proj"], #we only tape strips on the four towers, not every wall – saves compute.
    lora_dropout=0.05, # randomly erase 5 % of the strip during training to avoid memorization.
    bias="none",#we don’t touch the castle’s existing lights (biases); just the strips.
    task_type="CAUSAL_LM"
)


# Load model and tokenizer 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = quant_config,
    device_map="auto", #HuggingFace decides “these layers go on GPU 1, these on GPU 2…” if you have more than one, otherwise it just fills your single GPU.
    trust_remote_code=True,
    max_memory={0: "6GB"},  # Limit to 6GB VRAM - adjust based on your GPU
    offload_folder=None,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 3. ➜ ADD THIS to re-attach the PEFT wrapper to the same GPU
# model = model.cuda() 
# Inside every 4-bit layer it inserts little hooks that say:
# “When someone tries to update a weight, don’t touch the 4-bit brick – send the gradient to a side notebook instead.”
# This keeps the 4-bit bricks frozen but allows extra tiny matrices to be trained on top.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



# Dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def formatting_func(example):
    """Convert messages to Qwen chat format"""
    if "messages" in example:
        # Qwen models have a built-in chat template
        # This converts the messages list into the proper format
        text = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    else:
        raise ValueError("Dataset must have 'messages' field")

args = SFTConfig(
    output_dir= OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    optim="paged_adamw_8bit",#the optimizer’s own states (momentum, variance) are also 8-bit → another ~30 % memory saved.
    gradient_accumulation_steps=16,
    learning_rate=LEARNING_RATE,
    fp16=True,#do the multiply-adds in half precision; GPU happy, memory halved.
    max_grad_norm=0.3,#if a gradient vector gets too big, clip it (prevents explosion).
    weight_decay=0.001,#tiny L2 penalty so strips don’t grow unnecessarily large.
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    # dataset_text_field=None,
    # packing=True,#instead of padding every conversation to 1024, it concatenates conversations back-to-back with one special “reset” token → no wasted padding tokens → ~2× speed-up.
    max_seq_length=512,# if a single conversation is longer, it is cut; if shorter, more conversations are packed in.
    # If chat format (like yours with "messages"), leave None — trainer auto-applies chat template
    gradient_checkpointing=True,  # Save memory by recomputing activations
    gradient_checkpointing_kwargs={"use_reentrant": False},  # More stable
    dataloader_num_workers=0,  # Disable multiprocessing to save memory
    remove_unused_columns=True,
)

# Trainer
trainer = SFTTrainer(#Supervised Fine-Tune Trainer” (from TRL library)
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=args,
    formatting_func=formatting_func,
    packing=False
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done fine-tuning and saved model to", OUTPUT_DIR)