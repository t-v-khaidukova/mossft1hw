import io
import json
import os

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st


st.set_page_config(
    page_title="Fine-tuned MOSS SATS",
    page_icon="🎙️",
    layout="wide",
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_audio(uploaded_file):
    audio_bytes = uploaded_file.read()
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
    return audio, sample_rate, audio_bytes


def make_timeline(gold_segments, pred_segments, title="Speaker attribution"):
    """
    Expected segment format:

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
