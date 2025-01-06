# Multi-Agent AI Framework

This repository contains the Python implementation of a multi-agent AI framework based on the concepts discussed in the Medium article [Journey to Multi-Agent AI](https://medium.com/@juanestebanvenegas/journey-to-multi-agent-ai-949a05b61d39). The project leverages LangChain and LangGraph to create a robust multi-agent system for customer service and information retrieval tasks.

## Features

- **Supervisor Agent**: Manages user interactions and delegates tasks to specialized agents.
- **Customer Info Agent**: Retrieves customer-specific data using an SQLite database.
- **RAG Assistant**: Fetches general information using a vector store with embeddings for accurate responses.
- **Multi-Agent Graph**: Facilitates communication and context sharing between agents.
- **Memory Management**: Ensures persistent context using memory management tools.

## Project Structure

```
.
├── agent/                       # Agent logic and orchestration
│   ├── customer_info_agent.py   # Customer Info Agent
│   ├── rag_assistant.py         # RAG Assistant
│   └── supervisor_agent.py      # Supervisor Agent
├── data/                        # Data files
│   ├── source/                  # Raw data
│   └── processed/               # Processed data for vector store
├── text_embedding/              # Text embedding and vector store logic
│   ├── embedder.py              # Embedding generation and storage
│   └── vector_store.py          # Vector store management
├── services/                    # Core services
│   ├── db_service.py            # SQLite database services
│   ├── logging_config.py        # Logging configuration
│   ├── memory_manager.py        # Memory management logic
│   └── agent_setup.py           # Agent setup utilities
├── config/                      # Configuration files
│   └── settings.py              # Project settings
├── main.py                      # Entry point for running the graph
├── requirements.txt             # Required Python libraries
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your_username>/multi-agent-ai.git
   cd multi-agent-ai
   ```

2. Set up a Python virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add necessary configuration in `config/settings.py`.

## Usage

1. Ensure the required data files are in the `data/source/` directory and are processed into `data/processed/`.
2. Run the main application:

   ```bash
   python main.py
   ```

3. Interact with the system through the terminal or integrated UI.

## Requirements

See `requirements.txt` for the list of dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or support, reach out to [Juan Esteban Venegas](https://medium.com/@juanestebanvenegas).
