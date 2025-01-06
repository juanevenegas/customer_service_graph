import streamlit as st
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from graph.cs_graph import initialize_cs_graph  # Adjusted module import

# Initialize session state
if 'cs_graph' not in st.session_state:
    st.session_state.cs_graph = initialize_cs_graph()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'state' not in st.session_state:
    st.session_state.state = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Initialize state and config
def initialize_state_and_config(customer_id, thread_id):
    st.session_state.state = MessagesState(
        messages=[
            HumanMessage(content=f"customer_id:{customer_id}"),
        ]
    )
    st.session_state.config = {"configurable": {"thread_id": thread_id, "customer_id": customer_id}}
    st.session_state.initialized = True

# Process user input
def process_input():
    input_query = st.session_state.user_input.strip()  # Retrieve user input
    if input_query and st.session_state.config:  # Ensure there's input and config is initialized
        # Save user query to the chat history
        st.session_state.messages.append(("User", input_query))
        
        with st.spinner("Processing your query..."):  # Show a loading spinner
            try:
                # Recreate MessagesState with the initial customer_id message and the current user input
                customer_id = st.session_state.config["configurable"]["customer_id"]
                st.session_state.state = MessagesState(
                    messages=[
                        HumanMessage(content=f"customer_id:{customer_id}"),  # Always include the initial message
                        HumanMessage(content=input_query),  # Append the user input as a new message
                    ]
                )

                # Invoke the graph using the updated state and the existing config
                response = st.session_state.cs_graph.invoke(
                    st.session_state.state, 
                    config=st.session_state.config
                )

                # Extract the assistant's response
                assistant_response = response["messages"][-1].content

                # Save the assistant's response to the chat history
                st.session_state.messages.append(("Assistant", assistant_response))
            except Exception as e:
                # Handle any errors during graph invocation
                st.session_state.messages.append(("Assistant", f"Error: {str(e)}"))

        # Clear the input box for the next message
        st.session_state.user_input = ""

# Streamlit Interface
st.title("Customer Service Assistant (CS Graph)")

if not st.session_state.initialized:
    # Input for customer_id and thread_id
    with st.form("initial_inputs"):
        customer_id = st.text_input("Enter Customer ID:", key="customer_id")
        thread_id = st.text_input("Enter Thread ID:", key="thread_id")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if customer_id and thread_id:
                initialize_state_and_config(customer_id, thread_id)
                st.success(f"State and Config initialized with Customer ID: {customer_id} and Thread ID: {thread_id}")
            else:
                st.error("Both Customer ID and Thread ID are required.")
else:
    # Display chat history
    with st.container():
        for role, content in st.session_state.messages:
            if role == "User":
                st.markdown(
                    f"""<div style='text-align: right; color: white; background: #156082; padding: 8px; margin: 5px; border-radius: 8px;'>
                    <strong>{role}:</strong> {content}</div>""", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style='text-align: left; color: black; background: #EDEDED; padding: 8px; margin: 5px; border-radius: 8px;'>
                    <strong>{role}:</strong> {content}</div>""", unsafe_allow_html=True
                )

    # Auto-scroll to the bottom
    st.markdown("<div id='scroll-to-bottom'></div>", unsafe_allow_html=True)
    st.components.v1.html(
        """
        <script>
        var element = document.getElementById('scroll-to-bottom');
        element.scrollIntoView({behavior: 'smooth', block: 'end'});
        </script>
        """,
        height=0
    )

    # User input with Enter key
    st.text_input(
        "Type your message here:",
        key="user_input",
        on_change=process_input
    )
