from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# -------- CONFIG --------
VIDEO_ID = "Gfr50f6ZBvo"

# -------- LOAD TRANSCRIPT --------
def load_transcript(video_id):
    try:
        transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript_list = transcript_data.to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        print("❌ Transcripts are disabled for this video.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error while fetching transcript: {e}")
        return None

# -------- BUILD VECTOR STORE --------
def build_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store

# -------- INITIALIZE (Run Once) --------
# TRANSCRIPT = load_transcript(VIDEO_ID)
# VECTOR_STORE = build_vector_store(TRANSCRIPT) if TRANSCRIPT else None

# retriever = VECTOR_STORE.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 4}
# )

# -------- LLM --------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# -------- PROMPT TEMPLATE --------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

# -------- FORMAT DOCS --------
def format_docs(docs):
    """Converts retrieved docs to a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)

# -------- MAIN CHAIN (RunnableParallel) --------
# parallel_chain = RunnableParallel({
#     "context": retriever | RunnableLambda(format_docs),
#     "question": RunnablePassthrough()
# })

# main_chain = (
#     parallel_chain
#     | prompt
#     | model
#     | StrOutputParser()
# )

# -------- MAIN FUNCTION --------
def ask_question(question: str, transcript: str = None) -> str:
    if transcript is None:
        return "Transcript not provided."

    # Build vector store per transcript
    VECTOR_STORE = build_vector_store(transcript)
    retriever = VECTOR_STORE.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Chain (same as before)
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | model | StrOutputParser()
    return main_chain.invoke(question)

