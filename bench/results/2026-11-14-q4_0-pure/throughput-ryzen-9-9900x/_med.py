import json,glob,sys,statistics as st
for f in sorted(glob.glob(sys.argv[1])):
    d=json.load(open(f)); r={}
    for x in d["text"]: r.setdefault(x["words"],[]).append(x)
    out=[]
    for w,xs in sorted(r.items()):
        out.append("%s: %.0f/%.1f n=%d p=%d" % (w, st.median(v["prompt_per_second"] for v in xs), st.median(v["predicted_per_second"] for v in xs), min(v["predicted_n"] for v in xs), xs[0]["prompt_n"]))
    print(d["label"], " | ".join(out), "ECHEC" if not d["text"] else "")
