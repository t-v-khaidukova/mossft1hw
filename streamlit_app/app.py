"""Fine-tuned MOSS — speaker attribution and timestamps (SATS).

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

import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
GOLD = "#D4AF37"
PRED = "#4C78A8"

st.set_page_config(page_title="Fine-tuned MOSS SATS", page_icon="🎙️", layout="wide")


# --------------------------------------------------------------------- helpers

@st.cache_data
def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def make_timeline(gold_segments, pred_segments, title="Speaker attribution"):
    """Gold above, prediction below, one row per speaker.

    One Bar trace PER SOURCE, not per segment. The obvious version adds a trace
    for every segment, which on a 521-turn call means 521 traces: plotly then
    renders for several seconds and the legend has to be faked with dummy
    traces. Passing arrays keeps it at two traces however long the call is.
    """
    speakers = sorted({s["speaker"] for s in gold_segments}
                      | {s["speaker"] for s in pred_segments})
    if not speakers:
        return None
    row = {spk: i for i, spk in enumerate(speakers)}

    fig = go.Figure()
    for segs, label, colour, offset in ((gold_segments, "Gold (hand-marked)", GOLD, 0.18),
                                        (pred_segments, "Prediction", PRED, -0.18)):
        if not segs:
            continue
        fig.add_trace(go.Bar(
            x=[s["end"] - s["start"] for s in segs],
            base=[s["start"] for s in segs],
            y=[row[s["speaker"]] + offset for s in segs],
            orientation="h",
            width=0.28,
            name=label,
            marker=dict(color=colour, line=dict(width=0)),
            customdata=[[s["speaker"], s["start"], s["end"]] for s in segs],
            hovertemplate=(label + "<br>%{customdata[0]}"
                           "<br>%{customdata[1]:.2f}s → %{customdata[2]:.2f}s"
                           "<extra></extra>"),
        ))

    fig.update_layout(
        title=title,
        barmode="overlay",
        bargap=0,
        xaxis_title="Time (seconds)",
        yaxis=dict(tickmode="array", tickvals=list(row.values()),
                   ticktext=list(row.keys()), autorange="reversed"),
        height=max(260, 110 * len(speakers)),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


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


def local_model_dir():
    d = os.environ.get("MOSS_MODEL_DIR", "").strip()
    return d if d and Path(d).is_dir() else None


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
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype="auto"
    ).to(dtype).to(dev).eval()
    proc = AutoProcessor.from_pretrained(
        "OpenMOSS-Team/MOSS-Transcribe-Diarize", trust_remote_code=True
    )
    if not hasattr(proc, "feature_extractor"):
        raise RuntimeError("processor has no feature_extractor; it cannot encode audio")
    return model, proc, dev


def run_local(audio_bytes, model_dir):
    """Write to a temp file, generate, parse. Prompt must match training."""
    import tempfile

    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages, generate_transcription)

    prompt_file = Path(model_dir) / "PROMPT.txt"
    if not prompt_file.exists():
        raise RuntimeError(
            "PROMPT.txt is missing from the checkpoint. The model was trained to "
            "answer one exact instruction; sending a different one is a silent "
            "train/serve mismatch. Copy finetune_moss/prompt.txt in beside the "
            "weights."
        )
    prompt = prompt_file.read_text(encoding="utf-8").strip()

    model, proc, _dev = _load_local(model_dir)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(audio_bytes)
        path = fh.name
    try:
        text = generate_transcription(
            model, proc, build_transcription_messages(path, prompt),
            max_new_tokens=8192)
    finally:
        os.unlink(path)
    return parse_sats(text)


# ---------------------------------------------------------------------- header

st.title("🎙️ Fine-tuned MOSS — speaker attribution & timestamps")

st.markdown(
    """
**SATS** = speaker-attributed, time-stamped segmentation of two-party call audio.
This model answers **who spoke when**, and deliberately emits *no transcript*:

```
[0.87][S01][29.80]  [1.87][S02][2.40]  [3.22][S02][3.87]
```

Base model: `OpenMOSS-Team/MOSS-Transcribe-Diarize` (0.9B, Apache 2.0) —
a Whisper-Medium encoder and a Qwen3-0.6B decoder joined by a VQAdaptor.
Fine-tuned on ~599 h of in-domain Malaysian contact-centre audio.
"""
)

st.divider()

overview_tab, example_tab, results_tab, arch_tab, local_tab = st.tabs(
    ["📋 Overview", "📊 Example", "📈 Results", "🏗️ Architecture", "💻 Run it yourself"]
)


# -------------------------------------------------------------------- overview

with overview_tab:
    st.header("What this is, and what it is not")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
#### It does
- Segments a call into turns and attributes each to a speaker
- Emits start and end timestamps per turn
- Handles overlapping speech — backchannels spoken *while* the other
  party is talking, which is where a single-stream model usually fails

#### It does not
- Transcribe. There are no words in the output, by design.
- Identify *who* the speakers are — only that they are distinct
- Detect a third speaker reliably (warm transfers; see Architecture)
"""
        )
    with c2:
        st.markdown(
            """
#### Why no transcript
The target is ~5× shorter, which lets long audio fit the token budget and
removes the runaway-repetition failure mode. Capacity that went to
Malay/English/Mandarin orthography goes to speaker discrimination instead.

The base model **cannot be prompted** into this behaviour — given the same
instruction it transcribes anyway. It is entirely learned.

#### How it is measured
`speaker_timestamp_der` — the SGLang-Omni benchmark metric, computed by
upstream's own vendored code. **`cer` / `cp_cer` are undefined here**: there
is no transcript to score. They are reported as *n/a*, never as zero.
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
    st.header("Simulated example")
    st.caption(
        "A synthetic multi-speaker mixture, not a real call. Gold segments come "
        "from the simulator (exact by construction); predictions are "
        "pre-computed — no model runs in this app."
    )

    ex = load_json(ASSETS / "sim_example.json")
    wav = ASSETS / "sim_example.wav"

    if ex is None:
        st.warning(
            "`assets/sim_example.json` is missing. Generate it with "
            "`python3 make_assets.py` (see the repo README)."
        )
    else:
        if wav.exists():
            st.audio(str(wav))
        else:
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
            st.plotly_chart(fig, use_container_width=True)

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
        st.warning(
            "`assets/results.json` is missing. Generate it with "
            "`python3 make_assets.py`."
        )
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
              │                  onset .40 / offset .30
              │                  collar .02   (tuned)
              └─ ch2 (caller) ── energy VAD ─────────────── [S02]
                                        │
              ASR text ─────────────────┤  used ONLY to delete turns
              (never a training target) │  with no speech in them
                                        ▼
                            [start][Sxx][end] targets
                                        │
              + simulated mixtures (2–12 speakers) ─ the only >2-speaker signal
                                        │
                                        ▼
                  MOSS-Transcribe-Diarize, 1 epoch, ~599 h
                     Whisper-Medium enc → VQAdaptor → Qwen3-0.6B
                                        │
                                        ▼
              eval: DER + coverage against HAND-MARKED references only
""",
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
    st.header("Run it yourself")

    mdir = local_model_dir()

    if mdir is None:
        st.markdown(
            """
This app ships **no weights and no inference endpoint**. The fine-tuned model is
private, and a 0.9B model does not fit the hosted runtime (~1 GB RAM, no GPU)
even with credentials.

To run it for real, clone the repo on a machine with a GPU and point the app at
a local checkpoint:

```bash
export MOSS_MODEL_DIR=/path/to/outputs/ft2
streamlit run app.py
```

The weights are then read from disk and never leave the machine. There is no API
key anywhere in this app — nothing to store, nothing to leak.
"""
        )
        st.caption("`MOSS_MODEL_DIR` is not set, so inference is unavailable.")
    else:
        st.success(f"Local checkpoint detected: `{mdir}`")
        up = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
        if up is not None:
            audio_bytes = up.read()
            st.audio(audio_bytes)
            if st.button("Run speaker attribution", type="primary"):
                with st.spinner("Loading model and generating…"):
                    try:
                        segs = run_local(audio_bytes, mdir)
                    except Exception as exc:                       # noqa: BLE001
                        st.error(f"Inference failed: {exc}")
                    else:
                        st.success(f"{len(segs)} segment(s)")
                        st.dataframe(segs, use_container_width=True, hide_index=True)
                        fig = make_timeline([], segs, title="Predicted timeline")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "No transcript is produced. Timestamps and speaker "
                            "labels only."
                        )
# --------------------------------------------------------------------- helpers

@st.cache_data
def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def make_timeline(gold_segments, pred_segments, title="Speaker attribution"):
    """Gold above, prediction below, one row per speaker.

    One Bar trace PER SOURCE, not per segment. The obvious version adds a trace
    for every segment, which on a 521-turn call means 521 traces: plotly then
    renders for several seconds and the legend has to be faked with dummy
    traces. Passing arrays keeps it at two traces however long the call is.
    """
    speakers = sorted({s["speaker"] for s in gold_segments}
                      | {s["speaker"] for s in pred_segments})
    if not speakers:
        return None
    row = {spk: i for i, spk in enumerate(speakers)}

    fig = go.Figure()
    for segs, label, colour, offset in ((gold_segments, "Gold (hand-marked)", GOLD, 0.18),
                                        (pred_segments, "Prediction", PRED, -0.18)):
        if not segs:
            continue
        fig.add_trace(go.Bar(
            x=[s["end"] - s["start"] for s in segs],
            base=[s["start"] for s in segs],
            y=[row[s["speaker"]] + offset for s in segs],
            orientation="h",
            width=0.28,
            name=label,
            marker=dict(color=colour, line=dict(width=0)),
            customdata=[[s["speaker"], s["start"], s["end"]] for s in segs],
            hovertemplate=(label + "<br>%{customdata[0]}"
                           "<br>%{customdata[1]:.2f}s → %{customdata[2]:.2f}s"
                           "<extra></extra>"),
        ))

    fig.update_layout(
        title=title,
        barmode="overlay",
        bargap=0,
        xaxis_title="Time (seconds)",
        yaxis=dict(tickmode="array", tickvals=list(row.values()),
                   ticktext=list(row.keys()), autorange="reversed"),
        height=max(260, 110 * len(speakers)),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


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


def local_model_dir():
    d = os.environ.get("MOSS_MODEL_DIR", "").strip()
    return d if d and Path(d).is_dir() else None


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
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype="auto"
    ).to(dtype).to(dev).eval()
    proc = AutoProcessor.from_pretrained(
        "OpenMOSS-Team/MOSS-Transcribe-Diarize", trust_remote_code=True
    )
    if not hasattr(proc, "feature_extractor"):
        raise RuntimeError("processor has no feature_extractor; it cannot encode audio")
    return model, proc, dev


def run_local(audio_bytes, model_dir):
    """Write to a temp file, generate, parse. Prompt must match training."""
    import tempfile

    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages, generate_transcription)

    prompt_file = Path(model_dir) / "PROMPT.txt"
    if not prompt_file.exists():
        raise RuntimeError(
            "PROMPT.txt is missing from the checkpoint. The model was trained to "
            "answer one exact instruction; sending a different one is a silent "
            "train/serve mismatch. Copy finetune_moss/prompt.txt in beside the "
            "weights."
        )
    prompt = prompt_file.read_text(encoding="utf-8").strip()

    model, proc, _dev = _load_local(model_dir)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(audio_bytes)
        path = fh.name
    try:
        text = generate_transcription(
            model, proc, build_transcription_messages(path, prompt),
            max_new_tokens=8192)
    finally:
        os.unlink(path)
    return parse_sats(text)


# ---------------------------------------------------------------------- header

st.title("🎙️ Fine-tuned MOSS — speaker attribution & timestamps")

st.markdown(
    """
**SATS** = speaker-attributed, time-stamped segmentation of two-party call audio.
This model answers **who spoke when**, and deliberately emits *no transcript*:

```
[0.87][S01][29.80]  [1.87][S02][2.40]  [3.22][S02][3.87]
```

Base model: `OpenMOSS-Team/MOSS-Transcribe-Diarize` (0.9B, Apache 2.0) —
a Whisper-Medium encoder and a Qwen3-0.6B decoder joined by a VQAdaptor.
Fine-tuned on ~599 h of in-domain Malaysian contact-centre audio.
"""
)

st.divider()

overview_tab, example_tab, results_tab, arch_tab, local_tab = st.tabs(
    ["📋 Overview", "📊 Example", "📈 Results", "🏗️ Architecture", "💻 Run it yourself"]
)


# -------------------------------------------------------------------- overview

with overview_tab:
    st.header("What this is, and what it is not")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
#### It does
- Segments a call into turns and attributes each to a speaker
- Emits start and end timestamps per turn
- Handles overlapping speech — backchannels spoken *while* the other
  party is talking, which is where a single-stream model usually fails

#### It does not
- Transcribe. There are no words in the output, by design.
- Identify *who* the speakers are — only that they are distinct
- Detect a third speaker reliably (warm transfers; see Architecture)
"""
        )
    with c2:
        st.markdown(
            """
#### Why no transcript
The target is ~5× shorter, which lets long audio fit the token budget and
removes the runaway-repetition failure mode. Capacity that went to
Malay/English/Mandarin orthography goes to speaker discrimination instead.

The base model **cannot be prompted** into this behaviour — given the same
instruction it transcribes anyway. It is entirely learned.

#### How it is measured
`speaker_timestamp_der` — the SGLang-Omni benchmark metric, computed by
upstream's own vendored code. **`cer` / `cp_cer` are undefined here**: there
is no transcript to score. They are reported as *n/a*, never as zero.
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
    st.header("Simulated example")
    st.caption(
        "A synthetic multi-speaker mixture, not a real call. Gold segments come "
        "from the simulator (exact by construction); predictions are "
        "pre-computed — no model runs in this app."
    )

    ex = load_json(ASSETS / "sim_example.json")
    wav = ASSETS / "sim_example.wav"

    if ex is None:
        st.warning(
            "`assets/sim_example.json` is missing. Generate it with "
            "`python3 make_assets.py` (see the repo README)."
        )
    else:
        if wav.exists():
            st.audio(str(wav))
        else:
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
            st.plotly_chart(fig, use_container_width=True)

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
        st.warning(
            "`assets/results.json` is missing. Generate it with "
            "`python3 make_assets.py`."
        )
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
              │                  onset .40 / offset .30
              │                  collar .02   (tuned)
              └─ ch2 (caller) ── energy VAD ─────────────── [S02]
                                        │
              ASR text ─────────────────┤  used ONLY to delete turns
              (never a training target) │  with no speech in them
                                        ▼
                            [start][Sxx][end] targets
                                        │
              + simulated mixtures (2–12 speakers) ─ the only >2-speaker signal
                                        │
                                        ▼
                  MOSS-Transcribe-Diarize, 1 epoch, ~599 h
                     Whisper-Medium enc → VQAdaptor → Qwen3-0.6B
                                        │
                                        ▼
              eval: DER + coverage against HAND-MARKED references only
""",
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
    st.header("Run it yourself")

    mdir = local_model_dir()

    if mdir is None:
        st.markdown(
            """
This app ships **no weights and no inference endpoint**. The fine-tuned model is
private, and a 0.9B model does not fit the hosted runtime (~1 GB RAM, no GPU)
even with credentials.

To run it for real, clone the repo on a machine with a GPU and point the app at
a local checkpoint:

```bash
export MOSS_MODEL_DIR=/path/to/outputs/ft2
streamlit run app.py
```

The weights are then read from disk and never leave the machine. There is no API
key anywhere in this app — nothing to store, nothing to leak.
"""
        )
        st.caption("`MOSS_MODEL_DIR` is not set, so inference is unavailable.")
    else:
        st.success(f"Local checkpoint detected: `{mdir}`")
        up = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
        if up is not None:
            audio_bytes = up.read()
            st.audio(audio_bytes)
            if st.button("Run speaker attribution", type="primary"):
                with st.spinner("Loading model and generating…"):
                    try:
                        segs = run_local(audio_bytes, mdir)
                    except Exception as exc:                       # noqa: BLE001
                        st.error(f"Inference failed: {exc}")
                    else:
                        st.success(f"{len(segs)} segment(s)")
                        st.dataframe(segs, use_container_width=True, hide_index=True)
                        fig = make_timeline([], segs, title="Predicted timeline")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "No transcript is produced. Timestamps and speaker "
                            "labels only."
                        )

    {
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 3.42
    }
    """

    speakers = sorted(
        set(
            [s["speaker"] for s in gold_segments]
            + [s["speaker"] for s in pred_segments]
        )
    )

    speaker_to_y = {
        speaker: i for i, speaker in enumerate(speakers)
    }

    fig = go.Figure()

    # Gold = upper half of each speaker row
    for segment in gold_segments:
        speaker = segment["speaker"]

        fig.add_trace(
            go.Bar(
                x=[segment["end"] - segment["start"]],
                base=[segment["start"]],
                y=[speaker_to_y[speaker] + 0.18],
                orientation="h",
                width=0.28,
                name="Gold",
                marker=dict(color="#D4AF37"),
                hovertemplate=(
                    f"Gold<br>"
                    f"{speaker}<br>"
                    f"{segment['start']:.2f}s → "
                    f"{segment['end']:.2f}s"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Prediction = lower half
    for segment in pred_segments:
        speaker = segment["speaker"]

        fig.add_trace(
            go.Bar(
                x=[segment["end"] - segment["start"]],
                base=[segment["start"]],
                y=[speaker_to_y[speaker] - 0.18],
                orientation="h",
                width=0.28,
                name="Prediction",
                marker=dict(color="#4C78A8"),
                hovertemplate=(
                    f"Prediction<br>"
                    f"{speaker}<br>"
                    f"{segment['start']:.2f}s → "
                    f"{segment['end']:.2f}s"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Legend
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Gold",
            marker=dict(color="#D4AF37"),
        )
    )

    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Prediction",
            marker=dict(color="#4C78A8"),
        )
    )

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(speaker_to_y.values()),
            ticktext=list(speaker_to_y.keys()),
        ),
        height=max(350, 120 * len(speakers)),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def call_inference_api(audio_bytes, api_key):
    """
    Replace this with your actual inference API.

    The API should return something like:

    {
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.14
            }
        ]
    }
    """

    # TODO:
    # import requests
    #
    # response = requests.post(
    #     "https://YOUR-API-ENDPOINT/diarize",
    #     headers={"Authorization": f"Bearer {api_key}"},
    #     files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
    #     timeout=300,
    # )
    #
    # response.raise_for_status()
    # return response.json()

    raise NotImplementedError(
        "Connect call_inference_api() to your MOSS inference endpoint."
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🎙️ Fine-tuned MOSS SATS")

st.markdown(
    """
**Speaker Attribution and Timestamping (SATS)** using my fine-tuned MOSS model.

This release focuses on **who spoke when** — speaker attribution and
timestamps — rather than transcription.
"""
)

st.divider()


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

demo_tab, examples_tab, results_tab, architecture_tab = st.tabs(
    [
        "🎧 Demo",
        "📊 Examples",
        "📈 Results",
        "🏗️ Architecture",
    ]
)


# =========================================================
# DEMO
# =========================================================

with demo_tab:

    st.header("Run the model")

    st.write(
        "Upload an audio file and send it to the inference API."
    )

    api_key = st.text_input(
        "Inference API key",
        type="password",
        help="Used only to authenticate the inference request.",
    )

    uploaded_file = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
    )

    st.caption(
        "The hosted Streamlit app does not load the 0.9B model. "
        "Inference is performed by the separate API."
    )

    if uploaded_file is not None:

        audio, sample_rate, audio_bytes = load_audio(uploaded_file)

        st.audio(audio_bytes)

        duration = len(audio) / sample_rate

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Duration", f"{duration:.2f} s")

        with col2:
            st.metric("Sample rate", f"{sample_rate:,} Hz")

        if st.button(
            "Run speaker attribution",
            type="primary",
            disabled=not bool(api_key),
        ):

            with st.spinner("Running MOSS..."):
                try:
                    result = call_inference_api(
                        audio_bytes,
                        api_key,
                    )

                    st.success("Inference complete.")

                    st.subheader("Speaker segments")

                    segments = result.get("segments", [])

                    if segments:
                        st.dataframe(
                            segments,
                            use_container_width=True,
                        )

                    # Optional timeline
                    fig = make_timeline(
                        [],
                        segments,
                        title="Predicted speaker timeline",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                except Exception as exc:
                    st.error(f"Inference failed: {exc}")

    else:
        st.info(
            "Upload an audio file to run the model."
        )

    # Simulated example
    st.divider()

    st.subheader("Simulated example")

    sim_path = "assets/sim_example.wav"

    if os.path.exists(sim_path):

        st.audio(sim_path)

        if st.button("Use simulated example"):

            with open(sim_path, "rb") as f:
                sim_audio = f.read()

            st.session_state["sim_audio"] = sim_audio

            st.success(
                "Example loaded. Click Run speaker attribution "
                "to send it to the API."
            )


# =========================================================
# EXAMPLES
# =========================================================

with examples_tab:

    st.header("Real-call examples")

    st.write(
        """
These examples use **pre-computed predictions**. No model is loaded
and no audio is served for these examples.
"""
    )

    examples = {
        "Gold 58": "assets/gold58_example.json",
        "Long call": "assets/long_example.json",
    }

    example_name = st.selectbox(
        "Choose example",
        list(examples.keys()),
    )

    path = examples[example_name]

    if os.path.exists(path):

        data = load_json(path)

        gold = data.get("gold", data.get("gold_segments", []))
        prediction = data.get(
            "prediction",
            data.get("pred_segments", []),
        )

        fig = make_timeline(
            gold,
            prediction,
            title=example_name,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Gold segments are shown above predictions for each speaker. "
            "Aligned boundaries make agreement visually apparent."
        )

    else:
        st.warning(f"Missing example file: `{path}`")


# =========================================================
# RESULTS
# =========================================================

with results_tab:

    st.header("Results")

    results_path = "assets/results.json"

    if os.path.exists(results_path):

        results = load_json(results_path)

        if isinstance(results, list):
            st.dataframe(
                results,
                use_container_width=True,
            )

        elif isinstance(results, dict):

            for section_name, section_data in results.items():

                st.subheader(str(section_name))

                if isinstance(section_data, list):
                    st.dataframe(
                        section_data,
                        use_container_width=True,
                    )
                else:
                    st.write(section_data)

    else:
        st.warning(
            "`assets/results.json` has not been added yet."
        )


# =========================================================
# ARCHITECTURE
# =========================================================

with architecture_tab:

    st.header("Architecture")

    architecture_path = "assets/arch.png"

    if os.path.exists(architecture_path):
        st.image(
            architecture_path,
            use_container_width=True,
        )
    else:
        st.warning(
            "`assets/arch.png` has not been added yet."
        )

    st.subheader("Text-as-filter")

    st.markdown(
        """
The fine-tuned MOSS model is used for speaker attribution and
timestamp prediction.

For this release, the output is intentionally restricted to:

- **speaker identity**
- **segment start time**
- **segment end time**

The model output is therefore treated as a structured
speaker-timestamp representation rather than a transcript.
"""
    )
