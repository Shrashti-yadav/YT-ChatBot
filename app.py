import streamlit as st
from Backend.rag import ask_question, load_transcript

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥"
)

st.title("🎥 YouTube RAG Chatbot")
st.write("Ask questions from any YouTube video transcript.")

# -------- User Inputs --------
video_id = st.text_input("Enter YouTube Video ID:", "")
question = st.text_input("Ask a question:", "")

if st.button("Ask"):

    if not video_id.strip():
        st.warning("Please enter a video ID.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Loading transcript and generating answer..."):

            # 1️⃣ Load transcript dynamically via rag.py
            transcript = load_transcript(video_id)
            if not transcript:
                st.error("Transcript could not be loaded for this video.")
            else:
                # 2️⃣ Call RAG pipeline from rag.py
                answer = ask_question(question, transcript=transcript)
                st.success(answer)
