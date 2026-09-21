"""Property tests for stage 0 — the corpus streamer/splitter/cleaner.

No network and no downloads: every case is a string constructed here, so the
test pins the *behaviour* of the cleaning rules. Each assertion corresponds to a
failure mode that is actually present in the public TinyStories dump.
"""
import pathlib
import sys
from importlib import import_module

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ds = import_module("00_tinystories")

EOT = ds.EOT
STORY_A = ("Once upon a time, there was a little girl named Lily. She had a red ball. "
           "One day, she played with her friend Tom. They had fun together. "
           "Lily was happy.")
STORY_B = ("One day, a boy named Sam found a small dog in the park. The dog was cold and "
           "hungry. Sam took it home and gave it some food. The dog wagged its tail. "
           "Sam smiled.")


# 1. the official separator is a story boundary, always
docs = ds.split_stories(f"{EOT}\n{STORY_A}\n{EOT}\n\n{STORY_B}\n{EOT}")
assert len(docs) == 2, f"expected 2 stories, got {len(docs)}: {docs}"
assert docs[0].startswith("Once upon a time") and docs[1].startswith("One day")
print(f"[data] <|endoftext|> separated text -> {len(docs)} stories: OK")

# 2. a lost separator is recovered at the sentence boundary in front of an opener
glued = ("Once upon a time, there was a girl. She was happy. She liked to sing "
         "and dance in the garden every single day.Once upon a time, a boy had a "
         "kite. He flew it high in the sky until it was lost. He was sad.")
split = ds.split_stories(glued, min_chars=50, max_chars=2000)
assert len(split) == 2, f"glued stories not separated: {len(split)}"
print("[data] glued '...day.Once upon a time...' -> 2 stories: OK")

# 3. a blank line is NOT a story boundary (dialogue/moral are their own paragraph)
para = f"{STORY_A}\n\nThe girl learned that sharing is good.\n\nThe end."
out = ds.split_stories(para, min_chars=300, max_chars=2000)
assert len(out) == 1, f"blank line split one story into {len(out)}"
assert "The end." in out[0]
print("[data] paragraph breaks do not split a story: OK")

# 4. cleaning rejects each real failure mode, and keeps good prose
good = ds.clean_story(STORY_A, 100, 2500)
assert good is not None and good.endswith("happy."), "a clean story was rejected"
bad = {
    "too short": "Once upon a time, there was a cat.",
    "truncated mid-sentence": STORY_A[:-1] + " and then she",
    "url": STORY_A + " See https://example.com for more.",
    "chat marker": STORY_A + " <|endoftext|>",
    "prompt artifact": STORY_A + " Prompt GPT wrote this.",
    "lowercase start": "once upon a time, " + STORY_A[21:],
    "no sentence case": STORY_A + " she said yes. he said no.",
    "control character": STORY_A + "\x07bell",
    "digits only": "1 2 3 4 5 6 7 8 9 0 " * 20,
    "cjk wall": "Once upon a time. " + "汉" * 300,
}
for name, text in bad.items():
    assert ds.clean_story(text, 50, 2500) is None, f"clean_story accepted {name!r}"
print(f"[data] cleaner rejects all {len(bad)} failure modes, keeps clean prose: OK")

# 5. curly quotes and em dashes are kept (they are prose, not mojibake)
quoted = ds.clean_story(STORY_A[:-1] + ' "Well—maybe!" she said.', 100, 2500)
assert quoted is not None and "—" in quoted, "em dash / curly quote wrongly rejected"
print("[data] curly quotes + em dash survive the non-ASCII budget: OK")

# 6. mojibake repair is exact, and normalise is idempotent
assert ds.repair_mojibake("â€œhiâ€\x9d") == "“hi”", "mojibake map is wrong"
once = ds.normalise("A   story.\n\n\n\nNext.\tTabbed.")
assert once == ds.normalise(once), "normalise is not idempotent"
assert "  " not in once and "\n\n\n" not in once and "\t" not in once
print("[data] mojibake repair + idempotent normalisation: OK")

# 7. dedupe drops the second copy and nothing else; stats are consistent
stories, stats = ds.build_corpus(f"{EOT}{STORY_A}{EOT}{STORY_A}{EOT}{STORY_B}{EOT}",
                                 min_chars=100, max_chars=2500, max_chars_total=10 ** 7,
                                 log_every=0)
assert len(stories) == 2, f"dedupe kept {len(stories)} stories (expected 2)"
assert stats["stories"] == 2 and stats["source_stories"] == 3
assert stats["chars"] == sum(len(s) + len(ds.SEPARATOR) for s in stories)
print(f"[data] dedupe: 3 raw chunks -> {len(stories)} stories, stats consistent: OK")

# 8. the char budget is honoured, and the result is deterministic
many = EOT + EOT.join(f"{STORY_A} Story number {i}." for i in range(50)) + EOT
a, sa = ds.build_corpus(many, min_chars=100, max_chars=2500, max_chars_total=1500,
                        log_every=0)
b, sb = ds.build_corpus(many, min_chars=100, max_chars=2500, max_chars_total=1500,
                        log_every=0)
assert a == b and sa["chars"] == sb["chars"], "build_corpus is not deterministic"
assert all(len(s) <= 2500 for s in a), "a story exceeded max_chars"
print(f"[data] budget stop + determinism: {len(a)} stories, {sa['chars']:,} chars: OK")

print("all stage-0 tests passed")
