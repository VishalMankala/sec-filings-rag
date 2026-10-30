"""Streamlit UI:  streamlit run app.py"""

import streamlit as st

from secrag.config import COMPANIES, Settings
from secrag.pipeline import RAG

st.set_page_config(page_title="10-K Analyst", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading index and models...")
def load_rag(rerank: bool) -> RAG:
    return RAG.from_settings(Settings(), rerank=rerank)


st.title("10-K Analyst")
st.caption(f"Grounded answers with citations from the latest 10-K filings of {', '.join(COMPANIES)}.")

with st.sidebar:
    rerank = st.toggle("Cross-encoder reranking (slower; lower hit@1 on our eval)", value=False)
    st.markdown("**Try:**")
    examples = [
        "What does NVIDIA say about U.S. export controls?",
        "Compare how Apple and Microsoft describe their cybersecurity governance.",
        "What was Reality Labs' operating loss?",
        "What are Netflix's content obligations?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["q"] = ex

question = st.text_input("Ask about a company's 10-K", key="q")
if question:
    rag = load_rag(rerank)
    with st.spinner("Retrieving and answering..."):
        a = rag.ask(question)
    if a.refused:
        st.warning(a.text)
    else:
        st.markdown(a.text)
        if not a.citation_valid:
            st.error(f"Answer cited sources that do not exist: {a.invalid_citations}")
        if a.uncited_sentences:
            st.info(f"{len(a.uncited_sentences)} sentence(s) without a citation: verify before relying on them.")
    st.subheader("Sources")
    cited = set(a.cited)
    for n, h in enumerate(a.hits, 1):
        c = h.chunk
        label = f"[{n}] {c.ticker} FY{c.fiscal_year} · Item {c.item} {c.item_title}"
        label += "  ✓ cited" if n in cited else ""
        with st.expander(label, expanded=n in cited and n == min(cited, default=0)):
            st.write(c.text)
            st.markdown(f"[Open filing on EDGAR]({c.url})")
