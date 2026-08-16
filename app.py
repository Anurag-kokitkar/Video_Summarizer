"""
Streamlit UI for the AI Video Assistant.

Run with:
    streamlit run streamlit_app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
defaults = {
    "results": None,       # dict returned by run_pipeline
    "chat_history": [],    # list of (role, text) tuples
    "processing": False,
    "error": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --------------------------------------------------------------------------
# Pipeline runner (mirrors the CLI's run_pipeline, with progress reporting)
# --------------------------------------------------------------------------
def run_pipeline(source: str, language: str, progress_cb=None) -> dict:
    def step(msg, frac):
        if progress_cb:
            progress_cb(msg, frac)

    step("Loading and chunking media...", 0.10)
    chunks = process_input(source)

    step("Transcribing audio...", 0.30)
    transcript = transcribe_all(chunks, language=language)

    step("Generating title...", 0.50)
    title = generate_title(transcript)

    step("Summarizing...", 0.60)
    summary = summarize(transcript)

    step("Extracting action items...", 0.70)
    action_items = extract_action_items(transcript)

    step("Extracting key decisions...", 0.80)
    decisions = extract_key_decisions(transcript)

    step("Extracting open questions...", 0.90)
    questions = extract_questions(transcript)

    step("Building chat index...", 0.97)
    rag_chain = build_rag_chain(transcript)

    step("Done!", 1.0)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 Video Summarizer")
    st.caption("Transcribe, summarize, and chat with any video.")

    st.subheader("Source")
    input_mode = st.radio(
        "Input type", ["YouTube URL", "Upload file"], label_visibility="collapsed"
    )

    source = None
    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded = st.file_uploader(
            "Upload audio/video file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"]
        )
        if uploaded is not None:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source = tmp_path
            st.success(f"Ready: {uploaded.name}")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_clicked = st.button(
        "🚀 Run Pipeline", type="primary", use_container_width=True,
        disabled=st.session_state.processing,
    )

    if st.session_state.results:
        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.results = None
            st.session_state.chat_history = []
            st.session_state.error = None
            st.rerun()

# --------------------------------------------------------------------------
# Run pipeline on click
# --------------------------------------------------------------------------
if run_clicked:
    if not source:
        st.sidebar.error("Please provide a YouTube URL or upload a file.")
    else:
        st.session_state.processing = True
        st.session_state.error = None
        progress_bar = st.progress(0, text="Starting...")
        try:
            def update(msg, frac):
                progress_bar.progress(frac, text=msg)

            st.session_state.results = run_pipeline(source, language, progress_cb=update)
            st.session_state.chat_history = []
            progress_bar.empty()
        except Exception as e:
            st.session_state.error = str(e)
        finally:
            st.session_state.processing = False
        st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
if st.session_state.error:
    st.error(f"Something went wrong: {st.session_state.error}")

results = st.session_state.results

if not results:
    st.markdown(
        """
        ### Welcome 👋
        Paste a YouTube URL or upload a local audio/video file in the sidebar,
        then click **Run Pipeline** to get a transcript, summary, action items,
        key decisions, open questions — and a chat assistant for the video.
        """
    )
else:
    st.header(results["title"])

    tab_summary, tab_transcript, tab_actions, tab_decisions, tab_questions, tab_chat = st.tabs(
        ["📝 Summary", "📄 Transcript", "✅ Action Items", "📌 Key Decisions", "❓ Open Questions", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(results["summary"])

    with tab_transcript:
        st.text_area("Full transcript", results["transcript"], height=500)
        st.download_button(
            "Download transcript (.txt)",
            results["transcript"],
            file_name="transcript.txt",
        )

    with tab_actions:
        st.markdown(results["action_items"] or "_No action items found._")

    with tab_decisions:
        st.markdown(results["key_decisions"] or "_No key decisions found._")

    with tab_questions:
        st.markdown(results["open_questions"] or "_No open questions found._")

    with tab_chat:
        st.caption("Ask anything about the video's content.")

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        question = st.chat_input("Type your question...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(results["rag_chain"], question)
                    except Exception as e:
                        answer = f"Error answering question: {e}"
                st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))