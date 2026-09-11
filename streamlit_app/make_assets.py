#!/usr/bin/env python3
"""Build streamlit_app/assets/ from measured results. Run from the project root.

    python3 streamlit_app/make_assets.py

Ships NOTHING private:
  * results.json      numbers only, no audio, no transcripts
  * sim_example.*     a SIMULATED mixture -- synthetic multi-speaker audio built
                      from public speech corpora, not a customer call

The real evaluation sets are deliberately excluded. `test_ref_timed/` and
`test_long_ref_timed/` are hand-marked customer recordings carrying names,
phone, passport and IC numbers; their timelines would be publishable but their
audio is not, and shipping half of a pair invites the other half being asked
for. Numbers only.
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "assets"
SEG = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*\[(S\d+)\]\s*(.*?)\s*\[(\d+(?:\.\d+)?)\]", re.S)


def parse(text):
    out = []
    for a, spk, _t, b in SEG.findall(text or ""):
        a, b = float(a), float(b)
        if b > a:
            out.append({"speaker": spk, "start": round(a, 2), "end": round(b, 2)})
    return sorted(out, key=lambda s: (s["start"], s["end"]))


# ------------------------------------------------------------------ results

RESULTS = {
    "sections": [
        {
            "title": "THE SHIPPED MODEL — ft1 + hotwords, hand-marked short clips (58 calls, 1212.5 s)",
            "caption": "Every boundary placed by ear in Praat. The only set "
                       "verified on both text and time.",
            "rows": [
                {"config": "baseline MOSS (zero-shot, emits text)", "cer": "43.80%",
                 "DER @0.00": "17.30%", "miss": 12.49, "false alarm": 3.02,
                 "confusion": 1.79},
                {"config": "ft1 — 1 epoch on 649 h", "cer": "27.57%",
                 "DER @0.00": "13.50%", "miss": 3.98, "false alarm": 8.93,
                 "confusion": 0.59},
                {"config": "ft1 + hotwords  ← SHIPPED", "cer": "26.01%",
                 "DER @0.00": "12.59%", "miss": 3.23, "false alarm": 8.59,
                 "confusion": 0.77},
                {"config": "the energy VAD, scored as a model", "cer": "1.64%",
                 "DER @0.00": "13.95%", "miss": 0.57, "false alarm": 13.34,
                 "confusion": 0.04},
            ],
            "note": "The fine-tuned model beats the VAD labels it was trained "
                    "on (12.59% vs 13.95%) — and under conditions that flatter "
                    "the VAD, since the marking kit was anchored on its "
                    "boundaries, so ~a third scored zero error by construction.",
        },
        {
            "title": "THE SHIPPED MODEL — ft1 + hotwords, hand-marked long call (36.6 min, 521 turns)",
            "caption": "One call marked end to end: 567 windows, 1132 "
                       "boundaries, no sampling.",
            "rows": [
                {"config": "baseline MOSS (zero-shot, emits text)", "cer": "9.72%",
                 "DER @0.00": "11.01%", "miss": 3.97, "false alarm": 6.90,
                 "coverage": "95.9%", "coverage, turns <1 s": "72.7%"},
                {"config": "ft1 — 1 epoch on 649 h", "cer": "7.10%",
                 "DER @0.00": "9.64%", "miss": 1.62, "false alarm": 7.87,
                 "coverage": "98.2%", "coverage, turns <1 s": "89.7%"},
                {"config": "ft1 + hotwords  ← SHIPPED", "cer": "7.13%",
                 "DER @0.00": "9.40%", "miss": 1.63, "false alarm": 7.62,
                 "coverage": "98.2%", "coverage, turns <1 s": "95.6%"},
            ],
            "note": "What the fine-tune actually bought on long audio is "
                    "short-turn recall: baseline misses 27% of sub-second "
                    "speech, the fine-tune 4%. Those are backchannels spoken "
                    "while the other party is still talking — 60 s of 1664 s, "
                    "so DER can barely price it (0.83 points) even though it is "
                    "the difference a listener notices.",
        },
        {
            "title": "SEPARATE EXPERIMENT — text-free variant, simulated training only",
            "caption": "NOT the shipped model. A different fine-tune that emits "
                       "no words, trained on simulated mixtures alone (264 "
                       "optimizer steps, no real telephony). Its rows are not "
                       "comparable line-for-line with the two tables above: "
                       "different training data, different training budget, and "
                       "no transcript to score.",
            "rows": [
                {"config": "text-free variant — simulated test (in-domain)", "cer": "n/a",
                 "DER @0.00": "31.76%", "miss": 14.98, "false alarm": 5.21,
                 "confusion": 11.57},
                {"config": "text-free variant — hand-marked 58 (unseen)", "cer": "n/a",
                 "DER @0.00": "26.36%", "miss": 11.11, "false alarm": 5.88,
                 "confusion": 9.37},
            ],
            "note": "It scores no worse on real calls it has never heard than "
                    "on its own training domain — so this is undertraining, not "
                    "a transfer failure, and not evidence that dropping the "
                    "transcript hurts. Speaker confusion (9.37 vs 0.77) is the "
                    "dominant error and is what real two-speaker data should fix.",
        },
    ],
    "caveats": [
        "**Three models appear here and they are not interchangeable.** "
        "`baseline` is zero-shot MOSS; `ft1 + hotwords` is the shipped model "
        "and emits text; the `text-free variant` is a separate fine-tune with "
        "no transcript at all, so its `cer` reads *n/a*, never zero.",
        "**DER is not a timestamp metric.** It is time-weighted, so a "
        "half-second backchannel and a thirty-second turn are not "
        "comparably priced. Boundary quality is reported as median error "
        "and tolerance buckets instead.",
        "**Boundary error is bimodal** — 66% of turns land within 50 ms and "
        "~4% are out by seconds, so the mean falls in the empty valley "
        "between the two clusters. Quote the median.",
        "**The VAD rows of the two hand-marked sets are not comparable**: "
        "the marking kits were anchored on different VAD configurations, and "
        "an anchor scores ~0 on every boundary the annotator left alone.",
        "The benchmark metric *skips* samples it cannot parse, so a model "
        "that fails outright can outscore one that answers badly. Skip "
        "counts belong beside any DER.",
    ],
}


def pick_sim_example():
    """A simulated clip with a real prediction: moderate length, several
    speakers, and not one of the degenerate outputs."""
    pred = ROOT / "finetune_moss" / "pred_test.jsonl"
    if not pred.exists():
        return None
    best = None
    for line in pred.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        g, p = parse(r.get("gold", "")), parse(r.get("pred", ""))
        if not g or not p:
            continue
        dur = max(s["end"] for s in g)
        nspk = len({s["speaker"] for s in g})
        # reject collapsed predictions (zero-length loops) and text reversion
        if any(s["end"] - s["start"] <= 0 for s in parse(r.get("pred", ""))):
            continue
        if len(p) > 4 * len(g):
            continue
        if not (25 <= dur <= 70 and 3 <= nspk <= 6):
            continue
        score = abs(len(p) - len(g))
        if best is None or score < best[0]:
            best = (score, r["id"], g, p, dur)
    if best is None:
        return None
    _s, cid, g, p, dur = best
    return cid, g, p, dur


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(RESULTS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT / "results.json")

    got = pick_sim_example()
    if got is None:
        print("[warn] no usable simulated example found; skipping", file=sys.stderr)
        return 0
    cid, gold, pred, dur = got

    src = None
    for d in ("sim2_test", "sim2_long_test"):
        q = ROOT / "datasets" / d / "wav" / f"{cid}.wav"
        if q.exists():
            src = q
            break

    ex = {
        "title": "Simulated mixture — %d speakers, %.0f s" % (
            len({s["speaker"] for s in gold}), dur),
        "duration": round(dur, 2),
        "gold": gold,
        "prediction": pred,
        "note": "Gold here is the simulator's own ground truth, exact by "
                "construction. On real calls the reference is hand-marked and "
                "therefore itself uncertain — which is why the VAD's own "
                "boundaries had to be measured before any model was judged "
                "against them.",
        "licence": "Derived from public Malaysian speech corpora. Verify the "
                   "source licence before redistributing — some are CC-BY-NC, "
                   "which requires attribution and forbids commercial use.",
    }
    (OUT / "sim_example.json").write_text(
        json.dumps(ex, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT / "sim_example.json", f"({cid})")

    if src:
        shutil.copy2(src, OUT / "sim_example.wav")
        print("wrote", OUT / "sim_example.wav")
    else:
        print("[warn] audio for %s not found; the app will show the timeline "
              "only" % cid, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
