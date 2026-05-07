import streamlit as st
import os

from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from tavily import TavilyClient


# ---------------- API KEYS ----------------
api_key = st.secrets["MISTRAL_API_KEY"]
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])


# ---------------- TOOL 1: ARXIV ----------------
@tool
def research_papers(query: str) -> str:
    """Search arXiv papers based on query"""

    llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract short research keywords only. Return only keywords, no explanation."),
        ("human", "{query}")
    ])

    llm_response = llm.invoke(prompt.invoke({"query": query}))

    retriever = ArxivRetriever(top_k_results=5)
    docs = retriever.invoke(llm_response.content)

    return "\n\n".join([
        f"Title: {d.metadata.get('Title')}\n"
        f"Summary: {d.page_content}\n"
        f"Link: {d.metadata.get('Entry ID')}"
        for d in docs
    ])


# ---------------- TOOL 2: WEB / GITHUB ----------------
@tool
def web_recent_developments(query: str) -> str:
    """Search web for GitHub, blogs, project pages, implementations"""

    search_query = f"{query} github project blog implementation"

    results = tavily.search(
        query=search_query,
        max_results=5,
        search_depth="advanced",
        include_answer=True
    )

    if not results.get("results"):
        return "No web results found."

    output = []

    if results.get("answer"):
        output.append(f"Summary: {results['answer']}\n")

    for i, r in enumerate(results["results"], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")

        if "github.com" in url:
            source = "GitHub Repo"
        elif "arxiv.org" in url:
            source = "Paper"
        elif any(x in url for x in ["blog", "medium", "substack"]):
            source = "Blog"
        else:
            source = "Project Page"

        output.append(
            f"{i}. [{source}] {title}\n"
            f"{url}\n"
            f"{content[:200]}\n"
        )

    return "\n".join(output)


# ---------------- AGENT 1: PAPERS ----------------
@st.cache_resource
def get_paper_agent():
    llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key)

    return create_agent(
        llm,
        tools=[research_papers],
        system_prompt="""
You are a research paper assistant.
Always use research_papers tool first.
Explain results clearly.
Never hallucinate papers or links.
"""
    )


# ---------------- AGENT 2: WEB ----------------
@st.cache_resource
def get_web_agent():
    llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key)

    return create_agent(
        llm,
        tools=[web_recent_developments],
        system_prompt="""
You are a web research assistant.
Focus on:
- GitHub repos
- blogs
- project pages
- implementations

Summarize clearly.
"""
    )


paper_agent = get_paper_agent()
web_agent = get_web_agent()


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="📚")
st.title("📚 Multi-Agent Research Assistant")

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- SHOW HISTORY ----------------
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)


# ---------------- INPUT ----------------
query = st.chat_input("Ask a research question...")

if query:
    st.session_state.history.append(HumanMessage(content=query))

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):

            # Run both agents
            paper_result = paper_agent.invoke({
                "messages": st.session_state.history
            })

            web_result = web_agent.invoke({
                "messages": st.session_state.history
            })

            reply = f"""
📚 **ARXIV PAPERS**
{paper_result['messages'][-1].content}

🌐 **WEB / GITHUB / IMPLEMENTATIONS**
{web_result['messages'][-1].content}
"""

            st.write(reply)

    st.session_state.history.append(AIMessage(content=reply))