"""Fine-tuned MOSS — speaker-attributed, time-stamped transcription (SATS).

Showcases **ft1 + hotwords**: MOSS-Transcribe-Diarize fine-tuned for one epoch on
649 h of Malaysian contact-centre audio. It emits speaker, timestamps AND text.

A second, text-free variant (timestamps and speakers only) is a separate
experiment and is presented as such -- it is trained on simulated mixtures alone
and is not the shipped model. Keeping them apart matters: the shipped model has
a `cer` of 26.01%, which is meaningless for a model that emits no words, and
quoting one model's DER beside the other's CER would describe a system that does
not exist.

A SHOWCASE, not a live public demo. Two reasons, both structural:

  * The fine-tuned weights are private. A public app could only reach them with
    a token, and then that token is the only thing standing between the internet
    and the model.
  * Streamlit Community Cloud gives ~1 GB RAM and no GPU. The model is 1.8 GB in
    bf16. It cannot load there whatever the permissions say.

So the hosted app serves pre-computed outputs and never touches a checkpoint.
Set MOSS_MODEL_DIR to a local checkpoint and the "Run it yourself" tab does real
inference — on your machine, from weights that never left it.

No real customer audio is shipped. The example is a SIMULATED mixture; the
call-centre recordings this model was trained and measured on contain names,
phone, passport and IC numbers, and publishing them would be irreversible.

    streamlit run app.py
    MOSS_MODEL_DIR=/path/to/outputs/ft2 streamlit run app.py   # local inference
"""

import json
import os
from pathlib import Path

import altair as alt
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
GOLD = "#D4AF37"
PRED = "#4C78A8"

st.set_page_config(page_title="Fine-tuned MOSS SATS", page_icon="🎙️", layout="wide")


# --------------------------------------------------------------------- helpers

def load_json(path):
    """Read an asset, caching on CONTENT rather than on the path.

    The obvious `@st.cache_data def load_json(path)` caches a miss: if the file
    is absent on first load it returns None, and that None is then served for
    the rest of the session even after the file appears. That turns "asset not
    committed yet" into "asset still missing after I fixed it", which is a
    miserable thing to debug. Keying on mtime and size also means editing an
    asset refreshes it without clearing the cache by hand.
    """
    p = Path(path)
    if not p.exists():
        return None
    stat = p.stat()
    return _load_json_cached(str(p), stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def _load_json_cached(path, _mtime_ns, _size):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def missing_asset(name, how):
    """Say WHERE the app looked and WHAT it found. A bare "file is missing" sends
    you to check the repo, when the usual cause is that the app is looking
    somewhere else than you think, or the file was never committed."""
    st.warning("`assets/%s` was not found." % name, icon="⚠️")
    st.caption("Looked in: `%s`" % ASSETS)
    if ASSETS.is_dir():
        found = sorted(q.name for q in ASSETS.iterdir() if not q.name.startswith("."))
        st.caption("That directory exists and contains: %s"
                   % (", ".join("`%s`" % f for f in found) if found else "_nothing_"))
    else:
        st.caption("That directory **does not exist**. On Streamlit Cloud that "
                   "means `assets/` was never committed, or is excluded by a "
                   "`.gitignore` — `git check-ignore -v streamlit_app/assets/%s` "
                   "will say which rule." % name)
    st.caption(how)


def make_timeline(gold_segments, pred_segments, title="Speaker attribution"):
    """Gantt-style timeline: one row per (speaker, source).

    Altair, not plotly. Altair and Vega-Lite are core Streamlit dependencies, so
    the chart JS is whatever the installed Streamlit shipped with and cannot
    disagree with it. plotly is a separate package whose JS bundle has to match
    what the frontend expects, and when it does not the browser reports
    "TypeError: Importing a module script failed" -- a client-side error that no
    amount of Python testing reveals, and that an unpinned `plotly` in
    requirements.txt makes likely on any fresh deploy.

    Rows are labelled `S01 · gold` / `S01 · pred` rather than offset within one
    row: explicit categories need no `yOffset`, which is the other feature whose
    support varies by Vega-Lite version.
    """
    rows = []
    for segs, tag, label in ((gold_segments, "gold", "Gold (hand-marked)"),
                             (pred_segments, "pred", "Prediction")):
        for sg in segs:
            rows.append({
                "row": "%s · %s" % (sg["speaker"], tag),
                "speaker": sg["speaker"],
                "source": label,
                "start": float(sg["start"]),
                "end": float(sg["end"]),
                "dur": round(float(sg["end"]) - float(sg["start"]), 2),
            })
    if not rows:
        return None

    order = sorted({r["row"] for r in rows})
    height = max(160, 26 * len(order) + 60)

    return (
        alt.Chart(alt.Data(values=rows), title=title)
        .mark_bar(height=14, cornerRadius=2)
        .encode(
            x=alt.X("start:Q", title="Time (seconds)",
                    scale=alt.Scale(nice=False, zero=True)),
            x2="end:Q",
            y=alt.Y("row:N", title=None, sort=order),
            color=alt.Color(
                "source:N", title=None,
                scale=alt.Scale(domain=["Gold (hand-marked)", "Prediction"],
                                range=[GOLD, PRED]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[alt.Tooltip("source:N", title="Source"),
                     alt.Tooltip("speaker:N", title="Speaker"),
                     alt.Tooltip("start:Q", title="Start (s)", format=".2f"),
                     alt.Tooltip("end:Q", title="End (s)", format=".2f"),
                     alt.Tooltip("dur:Q", title="Duration (s)", format=".2f")],
        )
        .properties(height=height)
        .interactive(bind_y=False)
    )


def parse_sats(text):
    """`[start][Sxx][end]` or `[start][Sxx] words [end]` -> segment dicts.

    Upstream's own parser keeps only bare numbers and `S<n>` tokens and throws
    away whatever sits between brackets, which is exactly why a text-free model
    scores on `speaker_timestamp_der` unchanged. This mirrors that: text is
    read if present and ignored either way.
    """
    import re
    pat = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*\[(S\d+)\]\s*(.*?)\s*\[(\d+(?:\.\d+)?)\]",
                     re.S)
    out = []
    for a, spk, _txt, b in pat.findall(text or ""):
        a, b = float(a), float(b)
        if b > a:
            out.append({"speaker": spk, "start": a, "end": b})
    return sorted(out, key=lambda s: (s["start"], s["end"]))


PROMPT_FILE = Path(__file__).parent / "prompt_ft1hw.txt"


def ft1hw_prompt():
    """The exact instruction ft1+hotwords was measured with.

    ft1 and "ft1 + hotwords" are the SAME weights -- the hotwords are 18 domain
    proper nouns appended to the prompt, worth -1.5 CER on the verified set.
    Sending anything else is a different configuration, so this is read from a
    file rather than retyped.
    """
    if not PROMPT_FILE.exists():
        return None
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


@st.cache_data(show_spinner=False)
def _audio_data_uri(path, _mtime_ns, _size):
    import base64
    return "data:audio/wav;base64," + base64.b64encode(
        Path(path).read_bytes()).decode("ascii")


def audio_player(path):
    """st.audio, plus a plain-HTML fallback that cannot fail the same way.

    `st.audio` is a lazily-loaded frontend chunk. When the browser cannot fetch
    that chunk it reports "TypeError: Importing a module script failed" and the
    player simply is not there -- the same error `st.altair_chart` and
    `st.plotly_chart` give, which is the tell that the cause is Streamlit's
    static assets (a stale cache after a redeploy, an extension, a proxy) rather
    than any one charting library.

    The fallback is a bare <audio> tag with the file inline as a data URI: no
    module script, no network fetch, nothing to block. `unsafe_allow_html` is
    safe here because the content is a static asset from this repo, never user
    input -- but it costs ~1.3x the file size in page weight, so it stays behind
    an expander rather than being the default.
    """
    p = Path(path)
    if not p.exists():
        return False
    st.audio(str(p))
    with st.expander("Player not showing?"):
        st.caption(
            "Streamlit renders the player as a lazily-loaded script. If the "
            "browser could not fetch it, this plain HTML5 player will still "
            "work. A hard reload (⌘⇧R / Ctrl⇧R), an incognito window, or a "
            "different browser usually fixes the cause."
        )
        stat = p.stat()
        st.markdown(
            '<audio controls preload="metadata" style="width:100%%" src="%s">'
            "</audio>" % _audio_data_uri(str(p), stat.st_mtime_ns, stat.st_size),
            unsafe_allow_html=True,
        )
    return True


def degenerate(text, segs):
    """Flag the runaway output this model produces on non-speech input.

    Measured on ft1+hotwords with `no_repeat_ngram_size: 24` ACTIVE, which is
    not enough:

      digital silence -> "Sama-sama." x20, then "Sama." then "SAMA." -- the
                         block is evaded by changing the token form
      white noise     -> "[0.00][S01, 0.00][S02, ... [S38," -- malformed
                         brackets and 38 invented speakers
      50 Hz hum       -> timestamp and speaker-label garbage

    Worth catching because a public demo receives music, voice memos and silence,
    and a wall of repeated filler reads as a broken model rather than as
    out-of-distribution input. It emits no names or numbers in these cases -- the
    failure is embarrassing, not a disclosure.
    """
    import collections
    reasons = []
    t = (text or "").strip()
    if not t:
        return ["the model returned nothing"]
    rep = re.search(r"(.{3,40}?)\1{4,}", t)
    if rep:
        reasons.append("a short phrase repeats five or more times (`%s`)"
                       % " ".join(rep.group(1).split())[:40])
    spk = {g["speaker"] for g in segs}
    if len(spk) > 4:
        reasons.append("%d distinct speakers were emitted; these are two-party "
                       "calls" % len(spk))
    if re.search(r"\[S\d+\s*,", t):
        reasons.append("speaker tags are malformed (`[S01,` rather than `[S01]`)")
    if segs:
        zero = sum(1 for g in segs if g["end"] - g["start"] < 0.02)
        if zero > max(2, 0.2 * len(segs)):
            reasons.append("%d of %d segments have no duration" % (zero, len(segs)))
    words = re.findall(r"\w+", t.lower())
    if len(words) > 30:
        top, n = collections.Counter(words).most_common(1)[0]
        if n > 0.25 * len(words):
            reasons.append("the word `%s` is %.0f%% of the output"
                           % (top, 100.0 * n / len(words)))
    return reasons


def model_source():
    """(kind, value) -> ("dir", path) | ("repo", repo_id) | (None, None).

    A local directory for a GPU box; a PRIVATE Hugging Face repo for a hosted
    Space, where HF_TOKEN lives in the Space's secrets and the weights are
    pulled at boot. Either way the weights are never in this git repo.
    """
    d = os.environ.get("MOSS_MODEL_DIR", "").strip()
    if d and Path(d).is_dir():
        return "dir", d
    r = os.environ.get("MOSS_MODEL_REPO", "").strip()
    if r:
        return "repo", r
    return None, None


@st.cache_resource(show_spinner=False)
def _load_local(model_dir):
    """Load once per session. Processor comes from the PUBLIC base repo.

    A checkpoint written by moss-upstream/finetune.py is missing
    preprocessor_config.json and has had auto_map stripped from
    processor_config.json, so AutoProcessor on the checkpoint dir silently
    returns a text-only Qwen2Tokenizer with no feature_extractor -- every
    generation then comes back empty. Loading the processor from the base repo
    sidesteps that; repairing the checkpoint is the other fix.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch.autograd.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU bf16 is emulated and slow; float32 costs 3.6 GB of RAM and is faster.
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    token = os.environ.get("HF_TOKEN") or None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype="auto", token=token
    ).to(dtype).to(dev).eval()
    # The processor comes from the PUBLIC base repo either way -- see below.
    proc = AutoProcessor.from_pretrained(
        "OpenMOSS-Team/MOSS-Transcribe-Diarize", trust_remote_code=True
    )
    if not hasattr(proc, "feature_extractor"):
        raise RuntimeError("processor has no feature_extractor; it cannot encode audio")
    return model, proc, dev


def run_local(audio_bytes, model_dir, max_new_tokens=4096):
    """Write to a temp file, generate, parse. Prompt must match training."""
    import tempfile

    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages, generate_transcription)

    prompt = ft1hw_prompt()
    if prompt is None:
        raise RuntimeError(
            "prompt_ft1hw.txt is missing. ft1+hotwords IS ft1 plus that exact "
            "prompt; without it this would be plain ft1, which scores 1.5 CER "
            "points worse and is a different configuration."
        )

    model, proc, _dev = _load_local(model_dir)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(audio_bytes)
        path = fh.name
    try:
        raw = generate_transcription(
            model, proc, build_transcription_messages(path, prompt),
            max_new_tokens=max_new_tokens)
    finally:
        os.unlink(path)
    # generate_transcription may hand back a str or a dict depending on version
    text = raw if isinstance(raw, str) else (
        raw.get("text") or raw.get("transcript") or str(raw))
    return text, parse_sats(text)


# ---------------------------------------------------------------------- header

st.title("🎙️ Fine-tuned MOSS — speaker-attributed timestamped transcription")

st.markdown(
    """
**SATS** = *who spoke, when, and what they said* — as one string. The shipped
model is **ft1 + hotwords**, and it emits all three:

```
[2.61][S01] Thank you for calling. How may I assist you? [7.06]
[7.42][S02] Hi, I want to check my application status. [11.88]
```

Base model: `OpenMOSS-Team/MOSS-Transcribe-Diarize` (0.9B, Apache 2.0) —
a Whisper-Medium encoder and a Qwen3-0.6B decoder joined by a VQAdaptor.
Fine-tuned for one epoch on **649 h** of Malaysian contact-centre calls
(Malay / English / Mandarin, code-switched mid-sentence).

> A **text-free variant** — timestamps and speakers only, no words — is a
> separate experiment, shown under *Text-free variant*. It is not this model,
> and its numbers are not comparable line-for-line.
"""
)

st.divider()

overview_tab, results_tab, example_tab, arch_tab, local_tab = st.tabs(
    ["📋 Overview", "📈 Results", "🧪 Text-free variant", "🏗️ Architecture",
     "🎧 Run the model"]
)


# -------------------------------------------------------------------- overview

with overview_tab:
    st.header("What this is, and what it is not")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
#### It does
- **Transcribe** code-switched Malay / English / Mandarin telephony
- Attribute each turn to a speaker and time-stamp both edges
- Handle **overlapping speech** — the backchannels spoken *while* the other
  party is still talking, where a single-stream model usually fails

#### It does not
- Identify *who* the speakers are — only that they are distinct
- Detect a third speaker reliably (warm transfers; see Architecture)
- Emit word-level timestamps — 97.4% of words share a timestamp on these
  checkpoints, so only turn edges are usable
"""
        )
    with c2:
        st.markdown(
            """
#### What one epoch bought
On the hand-marked set, `cer` **43.80% → 26.01%** and DER **17.30% → 12.59%**.
It now beats the energy VAD that produced its own training labels (13.95%).

On a 36-minute call the gain is concentrated somewhere DER barely prices:
**sub-second turns go from 72.7% covered to 95.6%**. Those are backchannels
inside the other speaker's turn — 60 s of 1664 s, worth 0.83 DER points, and
the difference a listener actually hears.

#### Two decoding settings are required
`--max-new-tokens` scaled to clip length (~10 tokens per audio second) and
`no_repeat_ngram_size: 24`. Without them the model runs away on a few clips —
and neither fully removes it.
"""
        )

    st.info(
        "**No customer audio is published here.** The model was trained and "
        "evaluated on real call recordings containing names, phone, passport "
        "and IC numbers. Those stay private; the example below is a simulated "
        "mixture built from public speech corpora.",
        icon="🔒",
    )


# --------------------------------------------------------------------- example

with example_tab:
    st.header("Text-free variant — a separate experiment")

    st.warning(
        "**This is not the shipped model.** It is a second fine-tune that emits "
        "`[start][Sxx][end]` with no words at all, trained on **simulated "
        "mixtures only** — 264 optimizer steps, zero real telephony. Its DER is "
        "26–32% against the shipped model's 12.59%. Shown because the timeline "
        "below is the one example whose audio is safe to publish.",
        icon="🧪",
    )

    st.markdown(
        """
Why a text-free variant is worth trying: the target is ~5× shorter, so long
audio fits the token budget and the runaway-repetition failure has no words to
repeat. The base model **cannot be prompted** into it — given the same
instruction it transcribes anyway — so the behaviour is entirely learned.

Why its numbers are weak: it has seen no real calls. It scores *no worse* on
real telephony it has never heard (DER 26.36%) than on its own training domain
(31.76%), which is the signature of undertraining rather than a transfer
failure — and means **this is not evidence that dropping the transcript hurts**.
That question needs one variable changed on identical data, which has not been
run yet.
"""
    )

    st.subheader("Simulated example")
    st.caption(
        "A synthetic multi-speaker mixture, not a real call. Gold comes from the "
        "simulator (exact by construction); the prediction is pre-computed by "
        "the text-free variant — no model runs in this app."
    )

    ex = load_json(ASSETS / "sim_example.json")
    wav = ASSETS / "sim_example.wav"

    if ex is None:
        missing_asset("sim_example.json",
                      "Build it with `python3 streamlit_app/make_assets.py` from "
                      "the project root, then commit `streamlit_app/assets/`.")
    else:
        if not audio_player(wav):
            st.caption("_(audio not shipped — timeline only)_")

        gold = ex.get("gold", [])
        pred = ex.get("prediction", [])

        m = st.columns(4)
        m[0].metric("Duration", f"{ex.get('duration', 0):.1f} s")
        m[1].metric("Speakers (gold)", len({s["speaker"] for s in gold}))
        m[2].metric("Turns (gold)", len(gold))
        m[3].metric("Turns (predicted)", len(pred))

        fig = make_timeline(gold, pred, title=ex.get("title", "Simulated mixture"))
        if fig:
            st.altair_chart(fig, use_container_width=True)

        st.caption(
            "Gold sits above the prediction on each speaker row, so agreement "
            "and drift are visible directly. Hover for exact boundaries."
        )

        if ex.get("note"):
            st.info(ex["note"])

        if ex.get("licence"):
            st.caption(f"Audio licence: {ex['licence']}")


# --------------------------------------------------------------------- results

with results_tab:
    st.header("Results")

    res = load_json(ASSETS / "results.json")
    if res is None:
        missing_asset("results.json",
                      "Build it with `python3 streamlit_app/make_assets.py` from "
                      "the project root, then commit `streamlit_app/assets/`.")
    else:
        for section in res.get("sections", []):
            st.subheader(section.get("title", ""))
            if section.get("caption"):
                st.caption(section["caption"])
            if section.get("rows"):
                st.dataframe(section["rows"], use_container_width=True,
                             hide_index=True)
            if section.get("note"):
                st.info(section["note"])
            st.write("")

        if res.get("caveats"):
            st.subheader("How to read these")
            for c in res["caveats"]:
                st.markdown(f"- {c}")


# ---------------------------------------------------------------- architecture

with arch_tab:
    st.header("Architecture")

    img = ASSETS / "arch.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.code(
            """
stereo call ──┬─ ch1 (agent)  ── energy VAD ─────────────── [S01]
              │                  Schmitt-trigger hysteresis
              └─ ch2 (caller) ── energy VAD ─────────────── [S02]
                                        │
              per-channel ASR ──────────┤  supplies the TEXT, and deletes
              (a separate endpoint)     │  turns it heard nothing in
                                        ▼
                        [start][Sxx] text [end]  targets
                                        │
                                        ▼
                  MOSS-Transcribe-Diarize, 1 epoch, 649 h
                     Whisper-Medium enc → VQAdaptor → Qwen3-0.6B
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                 ft1 + hotwords                  text-free variant
                 SHIPPED                         (separate experiment)
                 [start][Sxx] text [end]         [start][Sxx][end]
                 cer 26.01%  DER 12.59%          cer n/a   DER 26-32%
                                        │
                                        ▼
              eval: DER + coverage against HAND-MARKED references only
"""
,
            language="text",
        )

    st.subheader("Where the labels come from")
    st.markdown(
        """
Each call is a stereo recording with **one party per channel**, so speaker
identity is free: a channel-0 slice cannot contain the caller. A per-channel
energy VAD with Schmitt-trigger hysteresis turns that into turn boundaries.

The VAD's thresholds were **tuned against hand-marked ground truth** — 580
boundaries on 58 short clips and 1,132 on one 36-minute call, all placed by ear
in Praat. That moved its bias from −0.114 / +0.140 s to ≈ 0.
"""
    )

    st.subheader("Text as a scaffold, not a target")
    st.markdown(
        """
The ASR transcript is used once, to drop VAD turns that contain no speech, and
then discarded. It is never a training target. That matters because the
alternative — training on text *and* timestamps — makes the words compete for
capacity with the task actually being learned.

⚠️ The filter reaches only turns where the ASR returned **nothing**. Turns where
it *invented* a short filler on silence (`"OK."`, `"herm."`) survive it — 4.8%
of hand-marked windows. Forced alignment was tested as a way to catch those and
**does not work on this corpus**: all source audio is 8 kHz narrowband against a
16 kHz aligner, and 20% of turns are pure digits, which the aligner's a–z
dictionary cannot represent.
"""
    )

    st.subheader("Known limits")
    st.markdown(
        """
- **Warm transfers.** When a second person takes over the caller's handset, no
  channel-derived label can see it. Speaker embeddings can, but embedding
  distance alone splits the *wrong* calls: over 10–36 minutes one voice drifts
  enough to cluster in two. The discriminator is temporal shape — a transfer is
  a handover with one changepoint, drift interleaves throughout.
- **Non-participant speech.** A colleague near the agent's mic reaches that
  channel and is labelled as the agent.
- **Length.** Training clips are capped at 10 minutes; longer audio is out of
  distribution.
"""
    )


# ------------------------------------------------------------- run it yourself

with local_tab:
    st.header("Run the model")

    kind, src = model_source()
    prompt = ft1hw_prompt()

    if kind is None:
        st.info(
            "**No model is configured, so this build serves results only.** "
            "That is the default for a public deployment: the weights are "
            "private and are never committed to this repo.",
            icon="🔒",
        )
        st.markdown(
            """
#### To run it for real

**On your own GPU box** — weights stay on local disk:

```bash
export MOSS_MODEL_DIR=/path/to/outputs/ft1
streamlit run app.py
```

**On a Hugging Face Space** — weights stay in a *private* model repo:

```
Space secrets:   HF_TOKEN        = a READ token for that repo
Space variables: MOSS_MODEL_REPO = YOUR-ORG/moss-sats-ft1
```

⚠️ **Streamlit Community Cloud cannot host this.** It gives ~1 GB of RAM and
no GPU; the weights alone are 1.82 GB. No token or setting changes that — the
process is killed before the model finishes loading. A Hugging Face Space on
the free CPU tier has **16 GB**, which fits.
"""
        )
    else:
        label = "local directory" if kind == "dir" else "private HF repo"
        st.success(f"Model configured — {label}: `{src}`")

        if prompt is None:
            st.error(
                "`prompt_ft1hw.txt` is missing. **ft1 + hotwords is ft1 plus "
                "that exact prompt** — the weights are identical. Without it "
                "this would silently be plain ft1, which is 1.5 CER points "
                "worse and a different configuration."
            )
        else:
            with st.expander("The exact prompt this sends (18 hotwords)"):
                st.code(prompt, language="text")

            st.caption(
                "Whatever you upload is processed in this app's own runtime and "
                "is not stored. On a CPU-only host expect **1–3 minutes** for a "
                "30-second clip; a GPU takes a few seconds."
            )

            up = st.file_uploader(
                "Upload a call recording",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                help="Stereo is fine — the model input is mono and is mixed down "
                     "automatically. Short clips first.",
            )

            if up is not None:
                audio_bytes = up.read()
                st.audio(audio_bytes)   # uploaded bytes; no fallback needed --
                # if this player is missing the page has a chunk-loading problem,
                # which the expander under the simulated example explains.
                mt = st.slider("Max new tokens", 512, 16384, 4096, step=512,
                               help="Roughly 10 tokens per second of audio. Too "
                               "low truncates the output and reads as a wall of "
                               "deletions rather than an error.")
                if st.button("Transcribe and attribute speakers", type="primary"):
                    with st.spinner("Loading the model and generating…"):
                        try:
                            text, segs = run_local(audio_bytes, src, mt)
                        except Exception as exc:                   # noqa: BLE001
                            st.error(f"Inference failed: {exc}")
                        else:
                            problems = degenerate(text, segs)
                            if problems:
                                st.error(
                                    "**This output is degenerate — the input is "
                                    "probably not a two-party call.** " +
                                    "; ".join(problems) + ".",
                                    icon="🌀",
                                )
                                st.caption(
                                    "The model was fine-tuned on stereo "
                                    "contact-centre recordings. On silence, "
                                    "music or a single-voice memo it runs away "
                                    "into repeated filler — measured, and "
                                    "`no_repeat_ngram_size: 24` does not stop "
                                    "it. Try a two-party conversation."
                                )
                            if not segs:
                                st.code(text[:2000] or "(empty)")
                            else:
                                st.success(f"{len(segs)} segment(s)")
                                fig = make_timeline([], segs,
                                                    title="Predicted speaker timeline")
                                if fig:
                                    st.altair_chart(fig, use_container_width=True)
                                st.subheader("Transcript")
                                st.dataframe(
                                    [{"speaker": g["speaker"],
                                      "start": round(g["start"], 2),
                                      "end": round(g["end"], 2)} for g in segs],
                                    use_container_width=True, hide_index=True)
                                with st.expander("Raw SATS output"):
                                    st.code(text, language="text")

    st.divider()
    st.caption(
        "**On reproducibility.** Decoding is greedy, but bf16 arithmetic is "
        "sensitive to dtype and attention-kernel choice, so a live run lands "
        "very close to the published numbers without being byte-identical — "
        "verified as agreeing for the first 107 characters of a clip before "
        "differing by 10 ms on one timestamp. Treat live output as "
        "representative, and the Results tab as the measurement."
    )
