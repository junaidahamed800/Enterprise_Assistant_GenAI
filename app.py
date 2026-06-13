import streamlit as st

from src_v2.p04_retriever import retrieve_context
from src_v2.p05_promptbuilder import build_prompt
from src_v2.p06_llm import generate_answer

st.set_page_config(
    page_title="Enterprise Assistant",
    page_icon="🤖",
)

st.title("🤖 Enterprise Assistant")

question = st.text_input(
    "Ask a question about company policies:"
)

if st.button("Submit"):

    if question:

        docs = retrieve_context(question)

        prompt = build_prompt(question, docs)

        answer = generate_answer(prompt)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for doc in docs:
                st.write(doc.page_content)