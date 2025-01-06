from langgraph.prebuilt import create_react_agent
from models.llm import llm
from langgraph.checkpoint.memory import MemorySaver
from models.vector_store import initialize_vector_store
from langchain.chains import load_summarize_chain
from langchain.tools import Tool
from langchain_core.messages import HumanMessage

###############################################################################
# Agent Retriever Tool
###############################################################################
vector_store = initialize_vector_store()

def retrieve(query: str, k: int = 3):
    """Retrieve and summarize information related to a query."""
    try:
        # Retrieve relevant documents
        retrieved_docs = vector_store.similarity_search(query, k=k)

        # Handle no results found
        if not retrieved_docs:
            return "No relevant information found.", "", []

        # Summarize retrieved documents if more than one is found
        if len(retrieved_docs) > 1:
            summarization_chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = summarization_chain.invoke(retrieved_docs)
        else:
            summary = retrieved_docs[0].page_content

        # Serialize metadata and content for detailed response
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return summary, serialized, retrieved_docs

    except Exception as e:
        # Error handling for unexpected issues
        return f"An error occurred: {str(e)}", "", []

retrieve_tool = Tool.from_function(
    func=retrieve,
    name="Retrieve",
    description="Retrieve and summarize relevant documents. The number of results can be adjusted with the 'k' parameter."
)

###############################################################################
# Agent Creation Function with Memory
###############################################################################
def create_rag_agent(llm=llm, checkpointer=MemorySaver()):

    ###########################################################################
    # Prompt
    ###########################################################################

    system_message = '''
    You are an intelligent agent specialized in retrieving and summarizing information. 

    1. Use the `retrieve_tool` to fetch relevant records related to the user's query. 
    2. Analyze the retrieved records and extract the most important and relevant details, including key metadata.
    3. Summarize the information clearly, concisely, and accurately.
    4. Always include sources or metadata to provide context when summarizing the results.
    5. If an error occurs during retrieval, inform the user about the issue without exposing technical details.
    6. Provide a well-structured response that addresses the user's question effectively, including follow-up recommendations if necessary.
    '''

    ###########################################################################
    # Tools
    ###########################################################################
    toolkit = [retrieve_tool]
    ###########################################################################
    # Agent creation
    ###########################################################################
    agent = create_react_agent(
        llm, 
        tools=toolkit,
        state_modifier=system_message,
        checkpointer=checkpointer
    )

    return agent


""" # Run a sample query
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
rag_agent = create_rag_agent()

thread_id = 1
customer_id = '1001'
config = {"configurable": {"thread_id": thread_id, "customer_id": customer_id}}
state= MessagesState(
    messages = [HumanMessage(content=f"what are BIMA working hours?")],
    thread_id=thread_id,
    customer_id=customer_id
    )

response = rag_agent.invoke(state, config=config)
response["messages"][-1].pretty_print()

"""
