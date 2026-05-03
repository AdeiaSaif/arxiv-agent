import streamlit as st


from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

import os

api_key = st.secrets["MISTRAL_API_KEY"]


# ---------------- TOOL ----------------
@tool
def research_papers(query: str) -> str:
    """Search arXiv papers based on query"""

    llm = ChatMistralAI(model="mistral-small-latest",  api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Extract short research keywords only.
Return only keywords, no explanation."""),
        ("human", "{query}")
    ])

    llm_response = llm.invoke(prompt.invoke({"query": query}))

    retriever = ArxivRetriever(top_k_results=5)
    docs = retriever.invoke(llm_response.content)

    return "\n\n".join(
        [
            f"Title: {d.metadata.get('Title')}\n"
            f"Summary: {d.page_content}\n"
            f"Link: {d.metadata.get('Entry ID')}"
            for d in docs
        ]
    )


# ---------------- AGENT ----------------
@st.cache_resource
def get_agent():
    llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key)

    return create_agent(
        llm,
        tools=[research_papers],
        system_prompt="""
You are a research assistant.

Always use research_papers tool first.
Then explain results simply and clearly.
Never hallucinate papers or links.
"""
    )


agent = get_agent()


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="ArXiv Research Agent", page_icon="📚")
st.title("📚 ArXiv Research Assistant")

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- SHOW CHAT HISTORY ----------
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)


# ---------- USER INPUT ----------
query = st.chat_input("Ask a research question...")

if query:
    # show user message
    st.session_state.history.append(HumanMessage(content=query))

    with st.chat_message("user"):
        st.write(query)

    # run agent
    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            result = agent.invoke({
                "messages": st.session_state.history
            })

            reply = result["messages"][-1].content
            st.write(reply)

    # update full history (IMPORTANT)
    st.session_state.history = result["messages"]