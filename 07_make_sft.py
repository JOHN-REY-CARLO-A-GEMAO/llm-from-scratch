"""Build a deterministic, small instruction-tuning corpus for the TinyStories LM.

The examples are synthetic and deliberately simple: greetings, animal facts,
everyday explanations, manners, multi-step arithmetic word problems, and tiny
stories.  JSONL keeps the prompt/response inspectable while ``text`` records the
exact chat serialization.  No network access or third-party package is needed.

## Chain-of-thought traces

Questions that need more than one step get a reasoning trace inside a
``<think> … </think>`` block before the answer:

    <|im_start|>assistant
    <think>
    1. Identify the core question: why does it rain.
    2. Check the key facts: water warms, rises into the sky, and cools into drops.
    3. Put the facts together and answer in one or two sentences.
    </think>
    Water warms, rises into the sky, and cools into tiny drops in clouds. …

Direct questions — greetings, one-fact answers, ``5 + 5`` — deliberately get *no*
trace, so the model learns *when* to reason instead of prefixing every reply with
a template. The exact shape is produced by ``personas.wrap_think`` (the same
function the renderer and the tests use), and the markers are ordinary bytes for
the byte-level BPE: the chat tokenizer's only added tokens stay
``<|im_start|>``/``<|im_end|>``.

## Document grounding (retrieval-augmented rows)

A few hundred rows carry a *system* turn built by `rag.context_block` — the same
function the engine calls at answer time — so the block the checkpoint is trained
on is byte-for-byte the block it is served:

    <|im_start|>system
    Context from uploaded document:
    [1] The zorble lives in the warm shallows of the Cora River. …

    Answer the user's question based ONLY on this context.<|im_end|>

The facts in those passages are invented (``zorble``, ``quibble``, ``Cora
River``), so "answer from the document" cannot be answered from memory instead.
Asking about a fact that is *not* in the passage is trained to
``The document does not say.`` — the honest answer, and the one that makes the
grounding score in the web UI meaningful rather than decorative.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:            # `personas` lives next to this file
    sys.path.insert(0, str(HERE))

import personas                         # noqa: E402  (stdlib-only, torch-free)
import rag                              # noqa: E402

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

#: Step wordings for reasoning traces. Several phrasings of the same three moves
#: stop the 7 M-parameter model from memorising one string and calling it
#: reasoning: the *shape* is the lesson, the words are incidental.
STEP_WORDS = (
    ("Identify the core question", "Check the key facts", "Put the facts together"),
    ("What is being asked", "What do I know", "Combine them"),
    ("Restate the question", "List the facts I know", "Connect the facts"),
    ("The question asks", "The facts I have", "So the answer is"),
    ("Step one: read the question", "Step two: gather the facts", "Step three: answer"),
)

DIRECT = 0          # an answer with no reasoning trace
REASONED = 1        # an answer preceded by a <think> block


def format_chat(prompt: str, response: str, system: str = "") -> str:
    """The chat serialization `generate.serialize_chat` produces, system turn and all."""
    block = f"{IM_START}system\n{system}{IM_END}\n" if system else ""
    return (f"{block}{IM_START}user\n{prompt}{IM_END}\n"
            f"{IM_START}assistant\n{response}{IM_END}")


class _Window:
    """Counts a rendered row in **chat tokens**, so the corpus fits the window.

    Character counts are not a proxy for this: BPE makes an 815-character row
    (256 tokens) *shorter* in tokens than a 636-character one (257), so the only
    authority on "does this fit" is the tokenizer itself. `08_sft.py` front-trims
    any row that overflows, and a trimmed row has lost part of its prompt or its
    injected document — it teaches the opposite of what it was written for. The
    tokenizer is a soft dependency: a checkout without `tokenizer/chat` still
    generates the corpus, it just cannot enforce the window.
    """

    def __init__(self, block_size: int = 256) -> None:
        self.block_size = block_size
        self.tok = None
        self.checked = 0
        self.dropped = 0
        try:
            import _stage
            self.tok = _stage.load("01_tokenizer").ByteLevelBPE.load(TOKENIZER_DIR)
        except Exception as exc:                 # pragma: no cover - env dependent
            print(f"[sft-data] no tokenizer at {TOKENIZER_DIR} ({exc}); "
                  f"rows are not checked against the {block_size}-token window")

    def fits(self, prompt: str, response: str, system: str = "") -> bool:
        if self.tok is None:
            return True
        self.checked += 1
        text = format_chat(prompt, response, system)
        if len(self.tok.encode_with_specials(text, [IM_START, IM_END])) <= self.block_size:
            return True
        self.dropped += 1
        return False


_WINDOW: "_Window | None" = None


def window() -> "_Window":
    """The process-wide row-length checker (built on first use, tokenizer once)."""
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = _Window()
    return _WINDOW


def cot(steps, answer: str) -> str:
    """Serialise a reasoning trace the way the corpus (and the UI) expects."""
    return personas.wrap_think(steps, answer)


def reasoned(question: str, facts: str, answer: str, variant: int = 0) -> str:
    """A three-step trace for an explanation-style question."""
    first, second, third = STEP_WORDS[variant % len(STEP_WORDS)]
    return cot([f"{first}: {question.strip().rstrip('?.').lower()}.",
                f"{second}: {facts.strip().rstrip('.')}.",
                f"{third} and answer in one or two short sentences."], answer)


def _greetings(rows) -> None:
    greetings = ["Hello!", "Hi there!", "Good morning!", "Good afternoon!", "Good evening!",
                 "How are you?", "Nice to meet you!", "Can we be friends?"]
    answers = [
        "Hello! I am happy to help you today.",
        "Hi! It is lovely to hear from you. What would you like to learn?",
        "Hello, friend! We can explore a question or make up a story together.",
    ]
    for i in range(200):
        rows.append((greetings[i % len(greetings)],
                     answers[(i // len(greetings)) % len(answers)], DIRECT))


def _animals(rows) -> None:
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
    questions = [
        ("Where does a {a} live?", "A {a} can live in {home}."),
        ("Tell me one fact about a {a}.", "Here is one fact: a {a} {fact}."),
        ("What does a {a} eat?", "A {a} eats {food}."),
        ("Can you explain a {a} to a child?", "A {a} is an animal that {fact}. You may find it in {home}."),
    ]
    for cycle in range(55):
        for a, (home, fact, food) in animals.items():
            q, ans = questions[cycle % len(questions)]
            rows.append((q.format(a=a), ans.format(a=a, home=home, fact=fact, food=food), DIRECT))


EXPLAIN = {
    "Why does it rain?": (
        "water warms, rises into the sky, and cools into tiny drops in clouds",
        "Water warms, rises into the sky, and cools into tiny drops in clouds. When the drops grow heavy, they fall as rain."),
    "Why is the sky blue?": (
        "sunlight has many colours and air scatters blue light more strongly than most others",
        "Sunlight has many colors. Air scatters blue light around the sky more strongly than most other colors, so the sky looks blue."),
    "What makes a rainbow?": (
        "sunlight bends and separates into colours when it passes through raindrops",
        "Sunlight bends and separates into colors when it passes through raindrops. Those colors make a rainbow."),
    "Why should I wash my hands?": (
        "soap and water remove dirt and many germs that can make people ill",
        "Washing removes dirt and many germs. Soap and water help keep you and other people healthy."),
    "Why do we sleep?": (
        "the body and the brain need time to rest, grow, and prepare for a new day",
        "Sleep gives the body and brain time to rest, grow, and prepare for a new day."),
    "How do plants grow?": (
        "plants use sunlight, water, air, and nutrients from the soil to make food",
        "Plants use sunlight, water, air, and nutrients from soil to make food and grow."),
    "What is a shadow?": (
        "light travels in straight lines, so an object blocks it and leaves a dark shape",
        "A shadow is a dark shape made when an object blocks light."),
    "Why do leaves fall?": (
        "some trees drop their leaves before cold or dry seasons to save water and energy",
        "Some trees drop their leaves before cold or dry seasons to save water and energy."),
    "How does a bicycle move?": (
        "your feet turn the pedals, the chain turns the back wheel, and the wheels roll",
        "Your feet turn the pedals. A chain turns the back wheel, and the rolling wheels move the bicycle forward."),
    "What is thunder?": (
        "lightning heats the nearby air very quickly and the air expands with a loud sound",
        "Lightning heats the nearby air very quickly. The air expands with a loud sound called thunder."),
    "Why do boats float?": (
        "a boat pushes aside water, and the water pushes back with enough force to hold it up",
        "A boat's shape pushes aside enough water to support its weight, so it floats."),
    "What is recycling?": (
        "used paper, glass, and metal can be collected and turned into new things",
        "Recycling turns used materials, such as paper, glass, or metal, into things that can be used again."),
}


def _explanations(rows) -> None:
    prefixes = ["", "Please explain: ", "Can you tell me: ", "I am curious. "]
    for cycle in range(55):
        for question, (facts, answer) in EXPLAIN.items():
            prefix = prefixes[cycle % len(prefixes)]
            prompt = prefix + (question[0].lower() + question[1:] if prefix else question)
            # Two thirds of the explanation set carries a trace: "why/how"
            # questions are exactly the ones that need facts lined up first.
            if cycle % 3 == 0:
                rows.append((prompt, answer, DIRECT))
            else:
                rows.append((prompt, reasoned(question, facts, answer, cycle), REASONED))


KINDNESS = [
    ("What should I do if a friend is sad?",
     "they may need someone to listen, and a trusted adult can help if the problem is big",
     "Listen kindly and ask if they want help. You can stay nearby, share a gentle word, and tell a trusted adult if they may be unsafe."),
    ("How can I share fairly?",
     "taking turns or splitting something equally feels fair to everyone",
     "Take turns or divide the item into equal parts. Ask everyone if the plan feels fair."),
    ("I made a mistake. What should I do?",
     "telling the truth and repairing the problem is what fixes a mistake",
     "Tell the truth, say sorry, and try to fix what happened. Mistakes can help us learn."),
    ("How do I welcome a new student?",
     "a smile and an invitation to join in helps someone feel included",
     "Smile, introduce yourself, and invite them to join an activity. A simple hello can help someone feel included."),
    ("What can I do when I feel angry?",
     "slow breathing and a quiet place help the feeling settle before you decide anything",
     "Pause and take slow breaths. Move to a calm place, name your feeling, and ask a trusted adult for help."),
    ("How can I be a good listener?",
     "looking at the speaker and waiting your turn shows you are paying attention",
     "Look toward the speaker, wait for your turn, and ask a kind question about what they said."),
]


def _kindness(rows) -> None:
    for i in range(600):
        question, facts, answer = KINDNESS[i % len(KINDNESS)]
        if i % 3 == 0:
            rows.append((question, reasoned(question, facts, answer, i // len(KINDNESS)), REASONED))
        else:
            rows.append((question, answer, DIRECT))


def _arithmetic(rows) -> None:
    """Exact single-step arithmetic. No trace: these are one fact, not a plan.

    (A bare ``5 + 5`` does not reach the model at all in the shipped chat UI —
    `tools.py` answers it. The examples stay in the corpus so the checkpoint
    still behaves when the tool hook is off.)
    """
    for repeat in range(3):
        for a in range(1, 21):
            for b in range(1, 11):
                forms = [f"What is {a} plus {b}?", f"Please add {a} and {b}.",
                         f"If I have {a} apples and get {b} more, how many do I have?"]
                rows.append((forms[repeat % 3], f"{a} plus {b} is {a+b}.", DIRECT))


def _plural(count: int, singular: str) -> str:
    """`2 boxes`, `1 box` — a corpus that says "1 pencils" teaches bad grammar."""
    if count == 1:
        return f"1 {singular}"
    suffix = "es" if singular.endswith(("s", "x", "z", "ch", "sh")) else "s"
    return f"{count} {singular}{suffix}"


NAMES = ["Mia", "Leo", "Ava", "Sam", "Nora", "Ben", "Lily", "Omar", "Zoe", "Max"]
THINGS = ["apples", "marbles", "stickers", "sweets", "books", "shells", "buttons"]
BOXES = ["box", "basket", "bag", "jar", "tin"]


def _word_problems(rows) -> None:
    """Two-step arithmetic word problems, with the arithmetic shown in the trace.

    This is the part of the corpus where the reasoning block earns its place: the
    trace *is* the working, and a reply that skips it gets the number wrong.
    """
    for i in range(450):
        name = NAMES[i % len(NAMES)]
        thing = THINGS[(i // 7) % len(THINGS)]
        a = 10 + (i * 7) % 11          # 10..20
        b = 2 + (i * 5) % 9            # 2..10
        c = 1 + (i * 3) % 8            # 1..8   (a + b - c stays positive)
        prompts = (
            f"{name} has {a} {thing}. {name} gets {b} more and then gives {c} away. "
            f"How many {thing} does {name} have now?",
            f"There are {a} {thing} in a basket. {name} adds {b} more, then takes out {c}. "
            f"How many {thing} are left?",
            f"{name} counted {a} {thing}, found {b} more, and gave {c} to a friend. "
            f"How many {thing} does {name} have?",
        )
        steps = [f"{name} starts with {a} {thing}.",
                 f"{name} gets {b} more: {a} + {b} = {a + b}.",
                 f"{name} gives {c} away: {a + b} - {c} = {a + b - c}."]
        rows.append((prompts[i % len(prompts)], cot(steps, f"{name} has {a + b - c} {thing} now."),
                     REASONED))

    for i in range(200):
        name = NAMES[(i * 3) % len(NAMES)]
        box = BOXES[(i // 4) % len(BOXES)]
        per = 3 + (i * 3) % 7          # 3..9 per container
        count = 2 + (i * 5) % 6        # 2..7 containers
        gone = 1 + (i * 2) % max(1, per * count - 1)
        noun = "crayons" if i % 2 else "pencils"
        containers = _plural(count, box)
        if i % 2:
            prompt = (f"Each {box} holds {per} crayons. There are {containers} and "
                      f"{_plural(gone, 'crayon')} {'is' if gone == 1 else 'are'} used. "
                      f"How many crayons are left?")
        else:
            prompt = (f"{name} has {containers} with {per} pencils in each {box}. "
                      f"{name} gives {_plural(gone, 'pencil')} away. How many pencils are left?")
        left = count * per - gone
        steps = [f"Each {box} holds {per} {noun}.",
                 f"{containers} hold {count} * {per} = {count * per} {noun}.",
                 f"Giving away {gone} leaves {count * per} - {gone} = {left}."]
        rows.append((prompt, cot(steps, f"{left} {noun} are left."), REASONED))


def _tool_word_problems(rows) -> None:
    """Multi-step problems whose arithmetic is done by the calculator hook.

    A 7 M-parameter model cannot add 16 and 6 reliably; the trace *can* say
    `<<calc: 16 - 6>>` and have the exact result substituted before the reply is
    shown (`tools.ToolTextFilter` does that inside the `<think>` block too). So
    the corpus teaches both halves of the trick: reason in steps, and let the
    tool do the arithmetic — which is exactly what the math tutor's hint says.
    """
    for i in range(300):
        name = NAMES[(i * 7) % len(NAMES)]
        thing = THINGS[(i // 5) % len(THINGS)]
        a = 12 + (i * 3) % 15          # 12..26
        b = 3 + (i * 5) % 9            # 3..11
        c = 1 + (i * 2) % 10           # 1..10  (a + b - c stays positive)
        total = a + b - c
        prompts = (
            f"{name} has {a} {thing} and gets {b} more. Then {name} gives {c} away. "
            f"How many {thing} does {name} have now?",
            f"A shop had {a} {thing}, sold {c}, then received {b} more. "
            f"How many {thing} are in the shop now?",
            f"{name} picked {a} {thing} on Monday and {b} on Tuesday, then shared {c}. "
            f"How many {thing} are left?",
        )
        steps = [f"{name} starts with {a} {thing}.",
                 f"Add the {b} more: <<calc: {a} + {b}>>.",
                 f"Take away the {c}: <<calc: {a} + {b} - {c}>>."]
        if i % 2:
            answer = f"{name} has <<calc: {a} + {b} - {c}>> {thing} now."
        else:
            answer = f"{name} has {total} {thing} now."
        rows.append((prompts[i % len(prompts)], cot(steps, answer), REASONED))


def _comparisons(rows) -> None:
    for i in range(300):
        a = 3 + (i * 5) % 17
        b = 1 + (i * 3) % 9
        c = 6 + (i * 7) % 20
        total = a + b
        question = f"Is {a} + {b} bigger than {c}?"
        steps = [f"Work out the sum first: {a} + {b} = {total}.",
                 f"Now compare {total} with {c}.",
                 f"{total} is {'larger' if total > c else 'smaller'} than {c}."]
        answer = (f"Yes. {a} + {b} = {total}, and {total} is bigger than {c}."
                  if total > c else
                  f"No. {a} + {b} = {total}, and {total} is smaller than {c}.")
        if i % 3 == 0:
            rows.append((question, answer, DIRECT))
        else:
            rows.append((question, cot(steps, answer), REASONED))

    sizes = [("two", 2), ("three", 3), ("five", 5), ("nine", 9), ("twelve", 12)]
    for i in range(120):
        words = " or ".join(name for name, _ in (sizes[i % 5], sizes[(i + 2) % 5]))
        big, small = sorted((sizes[i % 5], sizes[(i + 2) % 5]), key=lambda pair: pair[1])[::-1]
        steps = [f"The numbers to compare are {small[0]} and {big[0]}.",
                 f"{big[0]} is {big[1]} and {small[0]} is {small[1]}, so {big[0]} is larger.",
                 f"The larger number is {big[0]}."]
        rows.append((f"Which is bigger, {words}?",
                     cot(steps, f"{big[0].capitalize()} is bigger than {small[0]}."), REASONED))


#: The retrieval-augmented family. Every fact sentence names its subject and
#: carries the keyword the question asks with ("live"/"eat"/"make"), the values
#: are paired with names **at random** across rows, and the sentences are
#: shuffled inside the passage. That last part is the whole experiment: the first
#: version paired twelve fixed names with fixed facts, and the trained model
#: answered document questions from its prior — asked "Where does the marlow
#: live?" with a passage that plainly said *Low Fen*, it replied with the fact it
#: had memorized from training, and with an entity it had never seen it answered
#: "The document does not say." A mapping that cannot be memorized leaves
#: copying as the only way to be right, which is the skill document Q&A needs.
DOC_NAMES = (
    "zorble", "quibble", "grask", "marlow", "prell", "tove", "nibbin", "sundew",
    "kestrel", "hurn", "wispin", "glim", "otter", "bramble", "yarrow", "tarn",
    "spindle", "vellum", "cobb", "durn", "fennek", "pipit", "rowan", "sedge",
    "thistle", "umber", "verdin", "wicket", "yew", "zephyr",
)

DOC_HOMES = (
    "the warm shallows of the Cora River", "dry stone walls near the village of Marn",
    "the cool caves under Mount Ossa", "the tall grass of the Plain of Tesh",
    "the misty marshes of Low Fen", "the reed beds of Lake Vesh",
    "the chalk cliffs of Vire", "the quiet valley of Bran",
    "the pine woods above the town of Ilo", "the sandy dunes of Kest",
    "the hollow beneath the old oak", "the shadowy floor of the Amber Caves",
    "the rocky shore of the Isle of Renn", "the mossy banks of the river Ith",
    "the high meadows of Mount Sable", "the still water of the Mere",
    "the orchard behind the mill", "the frozen tundra of the far north",
    "the bracken on the heath", "the terraces of the Sunken Garden",
    "the mudflats at the mouth of the Cora", "the hollow logs by the waterfall",
    "the salt flats of Grim", "the wind-bent trees on the ridge",
    "the bramble thickets of the Weald", "the pebble beach at Orl",
    "the cave mouths of the Gray Hills", "the marsh grass of the eastern fen",
    "the deep leaf litter of the wood", "the cliff path above the harbour",
)

DOC_FOODS = (
    "small silver fish", "cactus flowers", "glowing moss", "bright red berries",
    "the bulbs of the reed flower", "sweet nectar", "amber seeds", "pale moon grass",
    "the bark of young willows", "wild honey", "the soft hearts of thistles",
    "fallen orchard apples", "the seeds of the sun-flower", "tiny water snails",
    "the leaves of the tea bush", "ripe cloudberries", "the roots of the marsh reed",
    "crisp beetles", "the flesh of the melon vine", "seaweed ribbons",
    "the buds of the rowan tree", "mushrooms from the forest floor",
    "the sap of the sugar maple", "winter moss", "the eggs of the reed warbler",
    "windfall plums", "the tips of young ferns", "hard black seeds",
    "the pollen of the meadow grasses", "small green caterpillars",
)

DOC_SOUNDS = (
    "a low humming whistle", "a sharp clicking rattle", "a deep ringing echo",
    "a soft fluting song", "a rumbling croak", "a thin whistling note",
    "a long mournful cry", "a bright chattering trill", "a hollow wooden knock",
    "a faint buzzing hum", "a steady ticking call", "a high piping squeak",
    "a gentle bubbling warble", "a harsh grating screech", "a clear bell-like tone",
    "a low throaty chuckle", "a slow rhythmic drumming", "a warbling whistle",
    "a soft hooting call", "a crackling chatter", "a mellow fluting note",
    "a shrill piping whistle", "a low mournful hoot", "a quick tapping beat",
    "a soft cooing murmur", "a rasping croak", "a delicate chiming song",
    "a deep booming call", "a light twittering song", "a hollow tapping knock",
)

#: One shape per field, keyword included, so the sentence to copy is easy to find
#: by keyword and impossible to guess by position (the sentences are shuffled).
#: `DOC_ALT_*` are rephrasings used by the *check*, not by training: a series no
#: row ever shows is how the report separates "copies" from "copies the exactly
#: one shape it saw".
DOC_HOME_SENT = "The {name} lives in {value}."
DOC_FOOD_SENT = "The {name} eats {value}."
DOC_SOUND_SENT = "The {name} makes {value}."
DOC_ALT_HOME = "You can find the {name} in {value}."
DOC_ALT_FOOD = "The {name} feeds on {value}."
DOC_ALT_SOUND = "The {name} calls out with {value}."

#: `(question, field)`. `field` is 0 home / 1 food / 2 sound, or None for "tell
#: me about it" (answer: the whole subject part of the passage, in passage order).
DOC_QUESTIONS = (
    ("Where does the {name} live?", 0),
    ("What does the {name} eat?", 1),
    ("What sound does the {name} make?", 2),
    ("Where can I find the {name}?", 0),
    ("What is the food of the {name}?", 1),
    ("What noise does the {name} make?", 2),
    ("Tell me about the {name}.", None),
)

#: Tokens reserved for the retrieved block inside the chat model's 256-token
#: window: enough for a five-sentence passage plus the instruction line, and it
#: is the same number the corpus, the check and the UI use.
DOC_CONTEXT_TOKENS = 168

#: Where the chat tokenizer lives: the window check above needs it, and it is the
#: same directory `08_sft.py` loads.
TOKENIZER_DIR = Path("tokenizer/chat")

#: Questions asked when the passage does **not** hold the answer: the honest
#: reply is the one trained here, not a guess from the model's prior.
DOC_MISSING = (
    ("How old is the {name}?", 0),
    ("What colour is the {name}?", 1),
    ("How many {name}s are left in the world?", 2),
    ("Who discovered the {name}?", 0),
    ("What does the {name} look like?", 1),
)


#: field -> sentence template. `0` home, `1` food, `2` sound; a question's field
#: is this index, and the answer is *that* sentence, in whatever order the
#: passage happens to hold it.
DOC_SENTENCES = (DOC_HOME_SENT, DOC_FOOD_SENT, DOC_SOUND_SENT)


def _subject_sentences(name: str, home: str, food: str, sound: str) -> dict[int, str]:
    values = (home, food, sound)
    return {field: template.format(name=name, value=value)
            for field, (template, value) in enumerate(zip(DOC_SENTENCES, values))}


def _passage(subject: dict[int, str], other: dict[int, str],
             i: int) -> tuple[str, list[tuple[int, str]]]:
    """Shuffle the subject's sentences into a passage that also holds a second
    creature's sentences, and return `(passage, subject_in_passage_order)`.

    The distractors are what make this a *search*: the answer sentence is never
    simply the first or the last line, and the other creature's sentence has the
    same shape, so position or shape alone cannot pick it out. `_passage` returns
    `(field, sentence)` pairs, not bare sentences — the rotation moves the
    sentences around, and a caller that indexes the *original* order asks about
    eating and answers with the sound.
    """
    fields = sorted(subject)
    order = [fields[(i + k) % len(fields)] for k in range(len(fields))]
    pairs = ([(field, subject[field]) for field in order]
             + [(field, other[field]) for field in sorted(other)[:2]])
    if i % 2:
        pairs = list(reversed(pairs))
    # `ordered` is the subject's pairs in the order the finished passage holds
    # them: "tell me about the X" answers with its three sentences, and reversing
    # the body without reversing the bookkeeping answered them in an order the
    # passage does not use.
    return " ".join(sentence for _, sentence in pairs), \
        [pair for pair in pairs if pair[0] in subject]


def _doc_rows(rows) -> None:
    """Retrieval-augmented rows: the passage answers the question, or it does not.

    The system text is produced by ``rag.context_block`` — the function the live
    engine uses — so a change to the injected format shows up in the corpus, the
    tests and the UI at the same time instead of in one of the three.
    """
    cost = lambda text: len(text) // 4 + 2          # noqa: E731  (~4 chars/token)
    count = len(DOC_NAMES)
    for i in range(2500):
        name = DOC_NAMES[(i * 7) % count]
        partner = DOC_NAMES[(i * 5 + 3) % count]
        subject = _subject_sentences(name,
                                     DOC_HOMES[(i * 11 + 3) % len(DOC_HOMES)],
                                     DOC_FOODS[(i * 13 + 5) % len(DOC_FOODS)],
                                     DOC_SOUNDS[(i * 17 + 7) % len(DOC_SOUNDS)])
        other = _subject_sentences(partner,
                                   DOC_HOMES[(i * 3 + 1) % len(DOC_HOMES)],
                                   DOC_FOODS[(i * 19 + 2) % len(DOC_FOODS)],
                                   DOC_SOUNDS[(i * 23 + 4) % len(DOC_SOUNDS)])
        body, ordered = _passage(subject, other, i)
        question, field = DOC_QUESTIONS[i % len(DOC_QUESTIONS)]
        prompt = question.format(name=name)
        if field is None:             # "tell me about it": every sentence, in order
            reply = " ".join(sentence for _, sentence in ordered)
        else:
            reply = next(sentence for asked, sentence in ordered if asked == field)
        # A trace on the single-sentence copies only. "Tell me about the X"
        # answers with three sentences, and three sentences plus a trace plus the
        # passage do not fit the 256-token window — the row would be front-trimmed
        # and lose part of its own document.
        traced = i % 3 == 0 and field is not None
        if traced:
            reply = cot([f"Find the sentence about the {name} in the context.",
                         "Copy that sentence out exactly as it is written."], reply)
        index = rag.DocumentIndex.from_text(body, name="notes.txt")
        system, _ = index.context_block(prompt, DOC_CONTEXT_TOKENS, cost, top_k=1)
        if not system:                    # budget too small: skip rather than mislead
            continue
        if not window().fits(prompt, reply, system):
            continue
        rows.append((prompt, reply, REASONED if traced else DIRECT, system))

    for i in range(250):
        # The passage mentions the subject — otherwise retrieval finds nothing and
        # the model is asked without a document at all. What it never contains is
        # the answer, which is the point of these rows.
        name = DOC_NAMES[(i * 7 + 4) % count]
        partner = DOC_NAMES[(i * 3 + 1) % count]
        missing, field = DOC_MISSING[i % len(DOC_MISSING)]
        subject = _subject_sentences(name,
                                     DOC_HOMES[(i * 11 + 3) % len(DOC_HOMES)],
                                     DOC_FOODS[(i * 13 + 5) % len(DOC_FOODS)],
                                     DOC_SOUNDS[(i * 17 + 7) % len(DOC_SOUNDS)])
        # keep only the fields that are *not* asked about, so the answer is absent
        other = _subject_sentences(partner,
                                   DOC_HOMES[(i * 3 + 1) % len(DOC_HOMES)],
                                   DOC_FOODS[(i * 19 + 2) % len(DOC_FOODS)],
                                   DOC_SOUNDS[(i * 23 + 4) % len(DOC_SOUNDS)])
        body, _ = _passage({k: v for k, v in subject.items() if k != field}, other, i)
        prompt = missing.format(name=name)
        index = rag.DocumentIndex.from_text(body, name="notes.txt")
        system, _ = index.context_block(prompt, DOC_CONTEXT_TOKENS, cost, top_k=1)
        if not system or not window().fits(prompt, "The document does not say.", system):
            continue
        rows.append((prompt, "The document does not say.", DIRECT, system))


def _stories(rows) -> None:
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
    prompts = ["Tell me a short story about a {c}.", "Can you make up a story about {n} and a {c}?",
               "Please tell a gentle story set in a {p}.", "Write a tiny bedtime story about a {c}."]
    for i in range(500):
        name = NAMES[i % len(NAMES)]
        creature = creatures[(i // 3) % len(creatures)]
        place = places[(i // 7) % len(places)]
        treasure = treasures[(i // 11) % len(treasures)]
        action, moral = lessons[(i // 13) % len(lessons)]
        prompt = prompts[i % len(prompts)].format(n=name, c=creature, p=place)
        response = (f"Once in a {place}, {name} met a little {creature} beside {treasure}. "
                    f"They wanted to {action}, so they listened, tried together, and soon began "
                    f"to smile. {moral}")
        rows.append((prompt, response, DIRECT))


def _persona_rows(rows) -> None:
    """Re-emit a slice of the corpus with each preset's system turn attached.

    The corpus is dominated by rows with **no** system turn, and the shipped
    checkpoints show what that costs: in front of a 40-token "you are a friendly
    assistant" block the 7 M model rambles through "why does it rain?" instead of
    planning, because a system block is a context it has only ever seen next to
    document questions. Repeating a sample of the *same* content with the same
    text `personas.resolve()` produces fixes the default path in both frontends
    — and it is the honest way to do it: the preset text is what the UI really
    sends, taken from the module that builds it, not a paraphrase.
    """
    presets = {slug: personas.resolve(slug, tool_hint=True)[1]
               for slug in ("friendly", "math", "reasoner")}
    plain = [row for row in rows if len(row) == 3]
    direct = [row for row in plain if row[2] == DIRECT and "<<calc:" not in row[1]]
    reasoned = [row for row in plain if row[2] == REASONED and "<<calc:" not in row[1]]
    marked = [row for row in plain if "<<calc:" in row[1]]

    def emit(pool, wanted: int, stride: int, system: str, mode: int) -> None:
        """Re-emit up to `wanted` rows with `system` in front, window permitting.

        A long story plus a persona block is the one combination that can overflow
        the window, and `emit` walks the pool rather than taking the first
        `wanted` entries: taking the head of the pool would *remove* long rows
        from the persona slice and teach that a system block means short answers.
        """
        made = 0
        step = 0
        limit = max(len(pool), wanted) * 8
        while made < wanted and step < limit:
            prompt, response, _ = pool[(step * stride) % len(pool)]
            step += 1
            if not window().fits(prompt, response, system):
                continue
            rows.append((prompt, response, mode, system))
            made += 1

    # Sizes are larger than the corpus's own share: `main()` samples an evenly
    # spaced subset when the pool exceeds `--count`, so the persona block has to
    # be over-emitted to survive the thinning at the rate it is wanted.
    emit(direct, 300, 7, presets["friendly"], DIRECT)      # tone rows
    emit(marked, 250, 3, presets["math"], REASONED)        # the calculator hint
    emit(reasoned, 250, 5, presets["reasoner"], REASONED)  # the reasoning hint


def examples() -> list[tuple]:
    """Every base example once: `(prompt, response, kind)`, kind is DIRECT or REASONED.

    Reaches the requested size by taking an evenly spaced subset when the family
    total is larger than `--count` (the default is 6,000), and by sampling
    deterministic repeats when it is smaller, so every build keeps every family
    and the two verification prompts.
    """
    rows: list[tuple] = [
        ("Hello! Who are you?", "Hello! I am a small, friendly assistant. I can answer simple questions and tell little stories.", DIRECT),
        ("Can you tell me a story about a frog?", "Once there was a little green frog named Pip. Pip was afraid to leap across a stream. A kind dragonfly said, ‘Try one small jump.’ Pip jumped, landed on a lily pad, and smiled. Soon he crossed the whole stream, one brave hop at a time.", DIRECT),
    ]
    for builder in (_greetings, _animals, _explanations, _kindness, _arithmetic,
                    _word_problems, _tool_word_problems, _comparisons, _stories,
                    _doc_rows):
        builder(rows)
    # last: it re-emits rows the builders above produced, so it must run after them
    _persona_rows(rows)
    return rows


def _row(row) -> tuple[str, str, int, str]:
    """Every base row as `(prompt, response, kind, system)` — system may be ""."""
    if len(row) == 4:
        return row
    prompt, response, kind = row
    return prompt, response, kind, ""


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
    # the requested size. This keeps every family in every build (and the two
    # verification prompts guaranteed); a count below the family total takes an
    # evenly spaced subset rather than truncating the last families away.
    if len(rows) > args.count:
        step = len(rows) / args.count
        pool = [rows[min(len(rows) - 1, int(i * step))] for i in range(args.count)]
        print(f"[sft-data] {len(rows):,} base rows -> evenly spaced subset of {args.count:,}")
    else:
        pool = list(rows)
        while len(pool) < args.count:
            pool.append(rows[rng.randrange(len(rows))])
    rng.shuffle(pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reasoned_rows = grounded_rows = 0
    with out.open("w", encoding="utf-8") as f:
        for row in pool:
            prompt, response, kind, system = _row(row)
            reasoned_rows += kind == REASONED
            grounded_rows += bool(system)
            json.dump({"prompt": prompt, "response": response,
                       "text": format_chat(prompt, response, system),
                       "system": system,
                       "reasoning": bool(kind == REASONED),
                       "grounded": bool(system)}, f, ensure_ascii=False)
            f.write("\n")
    print(f"[sft-data] wrote {len(pool):,} examples to {out} "
          f"({reasoned_rows:,} with a <think> trace, {grounded_rows:,} with a "
          f"document context, {len(rows):,} distinct base rows)")


if __name__ == "__main__":
    main()
