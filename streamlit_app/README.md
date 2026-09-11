# Fine-tuned MOSS — speaker attribution & timestamps

A Streamlit showcase for a fine-tuned
[`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
that segments two-party call audio into speaker-attributed, time-stamped turns
and emits **no transcript**:

```
[0.87][S01][29.80]  [1.87][S02][2.40]  [3.22][S02][3.87]
```

```bash
pip install -r requirements.txt
python3 make_assets.py          # run from the project root; builds assets/
streamlit run app.py
```

## This app ships no weights and no customer audio

Both are deliberate, and neither is a limitation to route around.

**The weights are private.** A public app could only reach them with a token,
and then that token is the only thing between the internet and the model. It
also would not help: Streamlit Community Cloud gives ~1 GB of RAM and no GPU,
and the model is 1.8 GB in bf16. It cannot load there whatever the permissions
say. So the hosted app serves **pre-computed** outputs and never touches a
checkpoint. There is no API key anywhere in it — nothing to store, nothing to
leak.

**The evaluation audio is real customer calls** containing names, phone,
passport and IC numbers. Publishing them is irreversible — clones and caches
survive deleting the repo — so only *numbers* derived from them are here. The
worked example is a **simulated** mixture built from public speech corpora.

## Running it for real

On a machine with the weights and a GPU:

```bash
export MOSS_MODEL_DIR=/path/to/outputs/ft2
streamlit run app.py
```

The "Run it yourself" tab then accepts an upload and runs actual inference. The
weights are read from local disk and never leave the machine.

Two requirements on that directory, both of which fail *silently* otherwise:

- **`PROMPT.txt` must sit beside the weights.** The model was trained to answer
  one exact instruction; sending a different one is a train/serve mismatch that
  produces plausible-looking wrong output rather than an error. The app refuses
  to run without it.
- **The processor is loaded from the public base repo, not the checkpoint.** A
  checkpoint written by `moss-upstream/finetune.py` lacks
  `preprocessor_config.json` and has had `auto_map` stripped from
  `processor_config.json`, so `AutoProcessor` on it returns a text-only
  `Qwen2Tokenizer` with no `feature_extractor` — and every generation comes back
  empty. The app sidesteps this; repairing the checkpoint is the other fix.

## Storing the model

Put the checkpoint in a **private** Hugging Face repo — it is ~2 GB, too large
for git, and a private repo gives you versioning without publication:

```bash
export HF_TOKEN=...            # a WRITE token; never pass it on a command line
python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("YOUR-ORG/moss-sats-ft2", private=True, exist_ok=True)
api.upload_folder(folder_path="outputs/ft2", repo_id="YOUR-ORG/moss-sats-ft2")
EOF
```

Include `PROMPT.txt` and a `PROVENANCE.md` recording the base model, the task,
the training data and **that only `speaker_timestamp_der` is meaningful** — the
`cer` family has no transcript to score and reads as total error.

## Licences

The base model is Apache 2.0, so a private derivative is permitted. The
simulated example derives from public Malaysian speech corpora; **verify the
source licence before redistributing** — some are CC-BY-NC, requiring
attribution and forbidding commercial use.

## Layout

```
app.py               the app; no weights, no keys, no customer audio
make_assets.py       builds assets/ from measured results
assets/
  results.json       the tables
  sim_example.json   pre-computed gold vs prediction for the simulated clip
  sim_example.wav    simulated audio (safe to publish)
  arch.png           optional; a text diagram is used if absent
```
