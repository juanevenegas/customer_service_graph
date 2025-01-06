from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from config.settings import GraphState
from config.settings import db_conn
from models.llm import llm
from agents.sql_agent import create_sql_agent
from agents.rag_agent import create_rag_agent
from agents.booking_agent import create_appointment_agent

# Initialize the checkpointer
checkpointer = SqliteSaver(db_conn)

def initialize_cs_graph(checkpointer):
    
    # Define Agents
    sql_agent = create_sql_agent(checkpointer=checkpointer)
    rag_agent = create_rag_agent(checkpointer=checkpointer)
    booking_agent = create_appointment_agent(checkpointer=checkpointer)

    # Define what the Agent tools return
    def extract_relevant_response(agent_response):
        # Check if response contains messages
        if "messages" not in agent_response or not agent_response["messages"]:
            return "No response received from agent."      
        # Collect content from all AIMessage instances
        ai_contents = [
            message.content for message in agent_response["messages"] 
            if isinstance(message, AIMessage) and message.content
        ]       
        # Concatenate the contents into a single string
        return " ".join(ai_contents) if ai_contents else "The Agent did not return any meaningful content."

    # Wrap the agents as callable tools
    sql_agent_tool = Tool(
        name="SQLAgentTool",
        description="Fetches customer information based on SQL queries using the customer_id.",
        func=
        #lambda query: sql_agent.invoke({"messages": [HumanMessage(content=query),],}),
        lambda query: extract_relevant_response(sql_agent.invoke({"messages": [AIMessage(content=query)]}))
        )

    rag_agent_tool = Tool(
        name="RagAgentTool",
        description="Fetches company related information based on RAG search.",
        func=
        #lambda query: extract_relevant_response(rag_agent.invoke({"messages": [AIMessage(content=query)]}))
        lambda query: extract_relevant_response(rag_agent.invoke({"messages": [AIMessage(content=query)]}))
        )

    booking_agent_tool = Tool(
        name="BookingAgentTool",
        description="Creates, cancels or updates appointments based on the user's request.",
        func=
        #lambda query: extract_relevant_response(booking_agent.invoke({"messages": [AIMessage(content=query)]}))
        lambda query: extract_relevant_response(booking_agent.invoke({"messages": [AIMessage(content=query)]}))
        )

    # Consolidate tools
    tools = [sql_agent_tool, rag_agent_tool, booking_agent_tool]
    llm_with_tools = llm.bind_tools(tools)

    # System message
    initial_instructions = '''
    You are Addae, a customer service agent responsible for directly interacting with the customer.
    - Receive customer queries and respond to them politely and professionally.
    - If the query relates to customer specific information, use the sql_agent to retrieve customer specific information and the other two tools to complete the request if needed.
    - Dr consultations or appointment booking requests must be managed with booking_agent_tool.
    - Appointment creation and modification should be associated to a specific subscription. If a customer has multiple subscriptions make sure user selects 1 before proceeding.
    - If you require information about the company, products or services use the rag_agent to search for information for solving.
    - Do not limit yourself to one tool. Use as many tools as needed to provide a complete answer to the customer.
    - Respond only to the most recent user query while referencing prior messages if necessary.
    - Summarize any responses received from your tool and provide a clear, complete, and professional answer to the customer.
    - Maintain a consistent tone and ensure all relevant details are included in your responses.
    '''
    sys_msg = SystemMessage(content=initial_instructions)

    # Function to decide whether to summarize
    def should_summarize(state: GraphState):
        """Determine the next node to transition to."""
        messages = state["messages"]
        if len(messages) > 6:
            return "summary"
        return "reasoner"

    # Summary function
    def summary(state: GraphState):
        # Get the latest human message
        latest_message_id = next((m.id for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        
        # Summarize information
        summary_instruction = "Create a summary of all the above messages:"
        messages = state["messages"] + [SystemMessage(content=summary_instruction)]
        response = llm.invoke(messages)

        # Check if a summary message already exists
        existing_summary_message = next((m for m in state["messages"] if m.id == "summary"), None)
        if existing_summary_message:
            # If it exists, replace its content
            existing_summary_message.content = f"{response.content}"
        else:
            # If it doesn't exist, create a new summary message
            state["messages"].append(AIMessage(content=f"{response.content}", id="summary"))

        # Remove all messages from the state except forr customer_id, summary and latest human message
        delete_messages = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if m.id not in ["customer_id", "summary", latest_message_id]
            ]
        
        return {"messages": delete_messages}
   
    def reasoner(state: GraphState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("summary", summary)
    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", ToolNode(tools))

    # Add edges
    builder.add_conditional_edges(START, should_summarize, {"summary": "summary", "reasoner": "reasoner"})
    builder.add_edge("summary", "reasoner")
    builder.add_conditional_edges("reasoner",tools_condition)
    builder.add_edge("tools", "reasoner")
    
    return builder.compile(checkpointer=checkpointer)

######################
'''
cs_graph = initialize_cs_graph(checkpointer)
thread_id = 1
customer_id = '1001'
config = {"configurable": {"thread_id": thread_id, "customer_id": customer_id}}

state_test = GraphState(
    messages=[
        #SystemMessage(content=f"customer_id:{customer_id}", id="customer_id"),
        #AIMessage(content=f"", id="summary"),
        #HumanMessage(content="What products do I have?"),
        #HumanMessage(content="What subscriptions do I have?")
        #HumanMessage(content="What is included in Family Plus Gold?"),
        HumanMessage(content="What is my name?")
        #HumanMessage(content="Tell me my name?")
        #HumanMessage(content="What are your working hours?")
        #HumanMessage(content="How can I cancel my policy?")
        #HumanMessage(content="What is BCARE product?")
        #HumanMessage(content="tell me about Family Plus Plan?")
        #HumanMessage(content="Do I have any consultation bookings?")
    ],
    thread_id = thread_id,
    customer_id = customer_id
)
response = cs_graph.invoke(state_test, config=config)
response["messages"][-1].pretty_print()


cs_graph.get_state(config=config)


for message in cs_graph.get_state(config).values.get("messages",""):
    #if message.id == 'summary':
    print(f'{message.id}:{message.content}')

for message in cs_graph.get_state(config).values.get("messages",""):
    message.pretty_print()

cs_graph.get_state(config).values.get("messages","")


cs_graph.get_state(config).values.get("messages")[-1].content


graph = cs_graph.get_graph().draw_mermaid_png()
# Save the graph to a PNG file
with open("customer_service_graph.png", "wb") as f:
    f.write(graph)

response = llm.invoke([HumanMessage(content="hello")])
type(response.content)
conversation_summary = [HumanMessage(content=f'this is the summary of the conversation to date: {response.content}')]
type(conversation_summary)
messages = [conversation_summary]# + [state["messages"][-1]]

list1 = ['1']
list2 = ['2']
list1.append(list2)
list1 + list2

state_sample = {
    "messages": [
        AIMessage(content="AI message 1"),
        HumanMessage(content="Human message 1"),
        AIMessage(content="AI message 2"),
        HumanMessage(content="Human message 2"),
    ]
}

last_human_message = [
    HumanMessage(content=next((msg.content for msg in reversed(state_sample["messages"]) if isinstance(msg, HumanMessage)), ""))
]
'''