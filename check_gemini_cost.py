import json, collections
tot = collections.Counter(); n = 0
for line in open(r"results/logs/final2/coordinator_calls.jsonl", encoding="utf-8"):
    r = json.loads(line); n += 1
    for k, v in r.items():
        if "token" in k.lower() and isinstance(v, (int, float)):
            tot[k] += v
print(n, "calls"); print(dict(tot))