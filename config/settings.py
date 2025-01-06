import os
import sqlite3
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import MessagesState
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# Define Graph State Class
class GraphState(MessagesState):
    thread_id: str
    customer_id: str

# Load environment variables from .env
load_dotenv()

# Access environment variables with fallback prompts
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Enter your OpenAI API Key: ").strip()

DATABASE_URI = os.getenv('DATABASE_URI')
if not DATABASE_URI:
    DATABASE_URI = input("Enter your Database URI: ").strip()

parsed_uri = urlparse(DATABASE_URI)
db_path = parsed_uri.path.lstrip('/')

# Setup langchain
LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
 
# Initialize db
db_conn = sqlite3.connect(db_path, check_same_thread=False)
db = SQLDatabase.from_uri(
    DATABASE_URI, 
    include_tables=[
        'customers',
        'customer_subscriptions',
        'subscription_payments',
        'customer_appointments'
    ],
    custom_table_info = {
            "customers": (
                "Table storing customer information. Columns include: "
                "id (Auto-increment unique identifier), "
                "customer_id (Unique identifier for customer), "
                "customer_name (First name of the customer), "
                "customer_last_name (Last name of the customer), "
                "customer_created_date (Date when customer was created)."
            ),

            "customer_subscriptions": (
                "Table storing customer subscriptions. Joins to customers table on customers.customer_id = customer_subscriptions.customer_id."
                "It is useful when fetching subscription or product information associated to a specific customer."
                "Columns include: "
                "id (Auto-increment unique identifier), "
                "customer_id (Unique identifier for customer), "
                "subscription_id (Unique identifier for subscription), "
                "subscription_start_date (Start date of the subscription), "
                "subscription_end_date (End date of the subscription), "
                "product_name (Name of the subscribed product, default 'tele doctor')."
            ),

            "subscription_payments": (
                "Table storing customer payment records."
                "This table does not include customer_id so the subscription_id must be obtained from customer_subscriptions table first to retrieve customer payment history"
                "Columns include: "
                "id (Auto-increment unique identifier), "
                "subscription_id (Unique identifier for subscription), "
                "payment_date (Date when the payment was made), "
                "amount_paid (Amount paid by the customer)."
            ),

            "customer_appointments": (
                "Table storing customer appointment records."
                "This table does not include customer_id so the subscription_id must be obtained from customer_subscriptions table first to retrieve customer payment history"
                "Columns include: "
                "id (Auto-increment unique identifier), "
                "subscription_id (Unique identifier for subscription), "
                "appointment_created_date (Date when the appointment was created), "
                "appointment_date (Scheduled date of the appointment), "
                "appointment_type (Type of appointment, e.g., general physician, specialist)."
            )
        }
    )