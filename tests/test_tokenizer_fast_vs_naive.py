"""Cross-check the fast incremental BPE against a simple reference implementation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import sys, random
from collections import Counter
from _stage import load
m = load("01_tokenizer")

def naive_train(texts, vocab):
    """Direct transcription of the textbook algorithm: full rescan every merge."""
    words = Counter()
    for t in texts:
        for mm in m.PAT.finditer(m.bytes_to_unicode(t.encode("utf-8"))):
            words[mm.group()] += 1
    sp = {w: list(w) for w in words}
    merges = []
    while len(merges) + 256 < vocab:
        pairs = Counter()
        for w, s in sp.items():
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += words[w]
        if not pairs:
            break
        best = min(pairs.items(), key=lambda kv: (-kv[1], "".join(kv[0])))[0]
        merges.append(best)
        a, b = best; new = a + b
        for w in sp:
            s = sp[w]; i = 0; n = len(s)
            while i < n - 1:
                if s[i] == a and s[i+1] == b:
                    s[i:i+2] = [new]; n -= 1; i += 1
                else:
                    i += 1
    return merges

random.seed(7)
alphabet = "abcde "
corpora = []
for _ in range(6):
    words = [ "".join(random.choice(alphabet) for _ in range(random.randint(1, 9)))
              for _ in range(random.randint(20, 120)) ]
    corpora.append(" ".join(words) * random.randint(1, 4))
corpora.append("the cat sat on the mat. the dog did not sit on the cat's mat!! " * 30)

ok = True
for i, text in enumerate(corpora):
    for vocab in (275, 295, 320):
        fast = m.ByteLevelBPE(vocab_size=vocab); fast.train([text], verbose=False)
        ref = naive_train([text], vocab)
        same = fast.merges == ref
        fast.check_invariants()   # per-word counts must match the words exactly
        print(f"corpus{i} vocab={vocab}: merges {'MATCH' if same else 'DIFFER'} ({len(fast.merges)} vs {len(ref)})")
        if not same:
            ok = False
            for j,(x,y) in enumerate(zip(fast.merges, ref)):
                if x != y:
                    print(f"   first divergence at merge {j}: fast={x} ref={y}"); break
print("RESULT:", "fast BPE == reference BPE on all corpora" if ok else "MISMATCH")
assert ok
# round-trip still holds after retraining
t = m.ByteLevelBPE(vocab_size=320); t.train([corpora[-1]], verbose=False)
for s in [corpora[-1][:200], "€€.txt 42\n\n\témoji 😀", ""]:
    assert t.decode(t.encode(s)) == s
print("round-trip on retrained tokenizer: OK")
