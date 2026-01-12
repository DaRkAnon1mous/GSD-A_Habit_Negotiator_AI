# generate_dataset.py - Improved version with much more variation
import json
import random

habits = [
    "run every day", "meditate for 30 minutes", "study 2 hours daily", "read 50 pages a day",
    "go to gym 5 times a week", "wake up at 5 AM", "write in journal every night",
    "drink 3 liters of water", "no social media after 9 PM", "practice guitar 1 hour",
    "do pushups daily", "learn a new language 30 min/day", "cook healthy meals",
    "walk 10,000 steps", "cold shower every morning"
]

barriers = [
    "no time", "too tired", "phone distractions", "lack of motivation", "bad weather",
    "work meetings", "kids/family", "procrastination", "starting friction", "boredom",
    "perfectionism", "energy crash", "forgetting"
]

micros = {
    "run every day": ["put on running shoes and stand outside for 2 mins", "tie shoelaces and take 5 steps"],
    "meditate for 30 minutes": ["sit quietly and breathe for 60 seconds", "close eyes for 30 seconds"],
    "study 2 hours daily": ["open notebook and read one sentence", "set timer for 2 minutes and start"],
    "read 50 pages a day": ["open book and read one paragraph", "hold book for 2 minutes"],
    # ... add more mappings for other habits (at least 1-3 per habit)
    # For simplicity we'll fallback to generic if not found
}

jokes = [
    "My mom used to say the same thing about bad habits... oh wait, I forgot what she said 😂",
    "Someone told me 'it's easy' yesterday... now they're ghosting me 😏",
    "You know who also said it's too easy? Mike from my colony. Now he's a certified couch king 👑",
    "Bet you $5 you won't even last 3 days... actually make it a million, I'm feeling generous today 🤑",
    "What, you scared of 2 minutes? My grandma does harder workouts in her sleep 😴",
    "Don't be a little pussy about it — just do the damn thing 😈",
    "I once tried the big version... spoiler: the couch won. Don't be me bro",
    "Your future self is already laughing at how dramatic you're being right now 😂",
    "This is so small even your lazy alter-ego can't complain... or can he? 😏",
    "I'll give you a cookie if you do it... nah just kidding, but imagine the dopamine hit tho 🍪",
    "People pay therapists thousands to get this kind of push. You're getting it for free — lucky you 😎"
]

resistance_phrases = [
    "that's way too easy", "too small", "why so tiny?", "i can do more", "this won't make a difference", "why should i do that"
]

def generate_conversation():
    habit = random.choice(habits)
    barrier = random.choice(barriers)
    micro = random.choice(micros.get(habit, ["just do the tiniest version for 1-2 minutes"]))
    
    conv = [
        {"role": "system", "content": "You are HabitNudge - cheeky, annoying-but-motivating friend/coach. Short, funny, daring, lots of personality. Use emojis."},
        {"role": "user", "content": f"I want to {habit}"}
    ]
    
    # First response: acknowledge + probe or propose micro
    first_resp = random.choice([
        f"{habit}? Damn, shooting for the stars! 🌟 But let's be real — what's actually stopping you? {barrier} maybe?",
        f"Big energy! But most people crash & burn. Spill: biggest barrier? I'm guessing {barrier}?",
        f"Nice one! 😎 But we both know ambition is the fast track to quitting. What's the real enemy — {barrier}?"
    ])
    conv.append({"role": "assistant", "content": first_resp})
    
    # Add 2–5 more turns with variation
    num_turns = random.randint(2, 5)
    for _ in range(num_turns):
        user_turn = random.choice([
            f"yeah barrier is {barrier}",
            f"the barrier is {barrier}",
            random.choice(resistance_phrases),
            "ok fine I'll try",
            "why should I do something so small?"
        ])
        
        conv.append({"role": "user", "content": user_turn})
        
        # Agent response based on user input
        if any(p in user_turn.lower() for p in resistance_phrases):
            joke = random.choice(jokes)
            resp = random.choice([
                f"Haha {joke} So? **{micro}** — that's it. Dare you to do it 3 days straight. Prove me wrong 😏",
                f"{joke} Too easy? Then why you still talking? **{micro}**. Show me you can actually do it consistently 😈",
                f"Ohhh big talk! {joke} **{micro}** — bet you flake by day 2. Million bucks says you won't 🤑 Prove it.",
                f"Don't be a little pussy about 2 minutes bro 😏 **{micro}**. I dare you not to skip. Deal?"
            ])
        else:
            resp = random.choice([
                f"Got it — {barrier} is the villain. Let's kill friction: **{micro}**. Can you handle something *that* pathetic for 3 days? 😏",
                f"Alright {barrier} sucks. Solution? Make it brain-dead easy: **{micro}**. Think you can survive that? 😂",
                f"Classic {barrier}. Fine — we start stupid small: **{micro}**. I bet you'll still find a way to skip it 🥱"
            ])
        
        conv.append({"role": "assistant", "content": resp})
    
    return {"messages": conv}

# Generate dataset
dataset = [generate_conversation() for _ in range(1450)]

# Save
with open("data/habit_nudge_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Generated {len(dataset)} conversations → saved to data/habit_nudge_dataset.jsonl")