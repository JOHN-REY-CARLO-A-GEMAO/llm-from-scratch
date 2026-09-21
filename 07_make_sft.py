"""Build a deterministic, small instruction-tuning corpus for the TinyStories LM.

The examples are synthetic and deliberately simple: greetings, animal facts,
everyday explanations, manners, arithmetic, and tiny stories.  JSONL keeps the
prompt/response inspectable while ``text`` records the exact chat serialization.
No network access or third-party package is needed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_chat(prompt: str, response: str) -> str:
    return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n{response}{IM_END}"


def examples() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Hello! Who are you?", "Hello! I am a small, friendly assistant. I can answer simple questions and tell little stories."),
        ("Can you tell me a story about a frog?", "Once there was a little green frog named Pip. Pip was afraid to leap across a stream. A kind dragonfly said, ‘Try one small jump.’ Pip jumped, landed on a lily pad, and smiled. Soon he crossed the whole stream, one brave hop at a time."),
    ]

    greetings = ["Hello!", "Hi there!", "Good morning!", "Good afternoon!", "Good evening!",
                 "How are you?", "Nice to meet you!", "Can we be friends?"]
    greeting_answers = [
        "Hello! I am happy to help you today.",
        "Hi! It is lovely to hear from you. What would you like to learn?",
        "Hello, friend! We can explore a question or make up a story together.",
    ]
    for i in range(360):
        g = greetings[i % len(greetings)]
        rows.append((g, greeting_answers[(i // len(greetings)) % len(greeting_answers)]))

    animals = {
        "frog": ("ponds and damp places", "jumps with strong back legs", "insects"),
        "rabbit": ("burrows and grassy fields", "hops quickly", "grass and leafy plants"),
        "elephant": ("grasslands and forests", "uses its trunk to smell, drink, and lift things", "plants"),
        "dolphin": ("the ocean", "swims and breathes air through a blowhole", "fish and squid"),
        "owl": ("forests and fields", "flies quietly and sees well in dim light", "small animals"),
        "penguin": ("cold coasts and islands", "swims with flipper-like wings", "fish"),
        "bee": ("gardens, meadows, and hives", "carries pollen between flowers", "nectar and pollen"),
        "turtle": ("oceans, ponds, or land, depending on its kind", "has a hard shell for protection", "plants or small animals"),
        "cat": ("homes and many places near people", "purrs when it feels calm", "meat-based food"),
        "dog": ("homes and many places near people", "uses a powerful sense of smell", "a balanced dog diet"),
        "giraffe": ("African grasslands", "uses its long neck to reach leaves", "leaves"),
        "butterfly": ("gardens and fields", "begins life as a caterpillar", "flower nectar"),
    }
    animal_questions = [
        ("Where does a {a} live?", "A {a} can live in {home}."),
        ("Tell me one fact about a {a}.", "Here is one fact: a {a} {fact}."),
        ("What does a {a} eat?", "A {a} eats {food}."),
        ("Can you explain a {a} to a child?", "A {a} is an animal that {fact}. You may find it in {home}."),
    ]
    for cycle in range(55):
        for a, (home, fact, food) in animals.items():
            q, ans = animal_questions[cycle % len(animal_questions)]
            rows.append((q.format(a=a), ans.format(a=a, home=home, fact=fact, food=food)))

    explain = {
        "Why does it rain?": "Water warms, rises into the sky, and cools into tiny drops in clouds. When the drops grow heavy, they fall as rain.",
        "Why is the sky blue?": "Sunlight has many colors. Air scatters blue light around the sky more strongly than most other colors, so the sky looks blue.",
        "What makes a rainbow?": "Sunlight bends and separates into colors when it passes through raindrops. Those colors make a rainbow.",
        "Why should I wash my hands?": "Washing removes dirt and many germs. Soap and water help keep you and other people healthy.",
        "Why do we sleep?": "Sleep gives the body and brain time to rest, grow, and prepare for a new day.",
        "How do plants grow?": "Plants use sunlight, water, air, and nutrients from soil to make food and grow.",
        "What is a shadow?": "A shadow is a dark shape made when an object blocks light.",
        "Why do leaves fall?": "Some trees drop their leaves before cold or dry seasons to save water and energy.",
        "How does a bicycle move?": "Your feet turn the pedals. A chain turns the back wheel, and the rolling wheels move the bicycle forward.",
        "What is thunder?": "Lightning heats the nearby air very quickly. The air expands with a loud sound called thunder.",
        "Why do boats float?": "A boat's shape pushes aside enough water to support its weight, so it floats.",
        "What is recycling?": "Recycling turns used materials, such as paper, glass, or metal, into things that can be used again.",
    }
    explain_prefixes = ["", "Please explain: ", "Can you tell me: ", "I am curious. "]
    for cycle in range(55):
        for q, ans in explain.items():
            prefix = explain_prefixes[cycle % len(explain_prefixes)]
            rows.append((prefix + (q[0].lower() + q[1:] if prefix else q), ans))

    kindness = [
        ("What should I do if a friend is sad?", "Listen kindly and ask if they want help. You can stay nearby, share a gentle word, and tell a trusted adult if they may be unsafe."),
        ("How can I share fairly?", "Take turns or divide the item into equal parts. Ask everyone if the plan feels fair."),
        ("I made a mistake. What should I do?", "Tell the truth, say sorry, and try to fix what happened. Mistakes can help us learn."),
        ("How do I welcome a new student?", "Smile, introduce yourself, and invite them to join an activity. A simple hello can help someone feel included."),
        ("What can I do when I feel angry?", "Pause and take slow breaths. Move to a calm place, name your feeling, and ask a trusted adult for help."),
        ("How can I be a good listener?", "Look toward the speaker, wait for your turn, and ask a kind question about what they said."),
    ]
    for i in range(650):
        rows.append(kindness[i % len(kindness)])

    # Exact arithmetic answers give the small model short, easy instruction patterns.
    for repeat in range(5):
        for a in range(1, 21):
            for b in range(1, 11):
                forms = [f"What is {a} plus {b}?", f"Please add {a} and {b}.",
                         f"If I have {a} apples and get {b} more, how many do I have?"]
                rows.append((forms[repeat % 3], f"{a} plus {b} is {a+b}."))

    names = ["Mia", "Leo", "Ava", "Sam", "Nora", "Ben", "Lily", "Omar", "Zoe", "Max"]
    creatures = ["frog", "rabbit", "fox", "turtle", "mouse", "duck", "owl", "kitten"]
    places = ["garden", "forest", "pond", "meadow", "little village"]
    treasures = ["a silver button", "a bright feather", "a tiny bell", "a smooth blue stone", "a golden leaf"]
    lessons = [
        ("share", "Sharing made the day brighter for everyone."),
        ("be brave", "Being brave meant trying one small step."),
        ("help a friend", "A kind helper can turn a hard task into an easy one."),
        ("tell the truth", "Telling the truth made the worry feel smaller."),
        ("be patient", "Good things can take a little time."),
    ]
    story_prompts = ["Tell me a short story about a {c}.", "Can you make up a story about {n} and a {c}?",
                     "Please tell a gentle story set in a {p}.", "Write a tiny bedtime story about a {c}."]
    for i in range(2100):
        n, c, p, treasure = names[i % len(names)], creatures[(i // 3) % len(creatures)], places[(i // 7) % len(places)], treasures[(i // 11) % len(treasures)]
        action, moral = lessons[(i // 13) % len(lessons)]
        prompt = story_prompts[i % len(story_prompts)].format(n=n, c=c, p=p)
        response = (f"Once in a {p}, {n} met a little {c} beside {treasure}. They wanted to {action}, "
                    f"so they listened, tried together, and soon began to smile. {moral}")
        rows.append((prompt, response))

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft/chat_sft.jsonl")
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if not 5000 <= args.count <= 10000:
        ap.error("--count must be between 5000 and 10000")
    rows = examples()
    rng = random.Random(args.seed)
    # Include every base example once, then sample deterministic repeats to reach
    # the requested size. This keeps the two verification prompts guaranteed.
    pool = rows[:args.count]
    while len(pool) < args.count:
        pool.append(rows[rng.randrange(len(rows))])
    rng.shuffle(pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for prompt, response in pool:
            json.dump({"prompt": prompt, "response": response,
                       "text": format_chat(prompt, response)}, f, ensure_ascii=False)
            f.write("\n")
    print(f"[sft-data] wrote {len(pool):,} examples to {out}")


if __name__ == "__main__":
    main()
