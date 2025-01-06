from graph.graph_setup import build_graph

# Simulate an authentication system
def authenticate_user():
    # Simulate fetching customer_id and thread_id from an external system
    customer_id = input("Enter authenticated Customer ID: ")
    thread_id = input("Enter active Thread ID: ")
    return customer_id, thread_id

def main():
    # Authenticate the user
    customer_id, thread_id = authenticate_user()

    # Build the graph
    graph = build_graph()

    print("Authentication successful. Chat service is ready! Type 'exit' to quit.\n")

    # Run the main conversation loop
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Pass customer_id and thread_id as metadata
        response = graph.run(
            user_input,
            config={"configurable": {"customer_id": customer_id, "thread_id": thread_id}}
        )
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()

