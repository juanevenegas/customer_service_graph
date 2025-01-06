from datetime import datetime, timedelta, timezone
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from models.llm import llm
from config.settings import db
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

###############################################################################
# Current DateTime Manager
###############################################################################

class DateTimeManager:
    _timezone = timezone.utc  # Default timezone set to UTC

    @staticmethod
    def set_timezone(tz):
        """Set the timezone for the DateTimeManager."""
        DateTimeManager._timezone = tz

    @staticmethod
    def now():
        """Returns the current datetime in the configured timezone."""
        return datetime.now(tz=DateTimeManager._timezone)

    @staticmethod
    def iso_now():
        """Returns the current datetime in ISO format and configured timezone."""
        return datetime.now(tz=DateTimeManager._timezone).isoformat()

###############################################################################
# Agent Creation Function with Appointment Management
###############################################################################

# Define input models
class CheckAppointmentsInput(BaseModel):
    subscription_id: str = Field(..., description="Subscription ID associated with the customer.")

class AppointmentInput(BaseModel):
    subscription_id: str = Field(..., description="Subscription ID associated with the appointment")
    appointment_date: str = Field(..., description="Scheduled date of the appointment in ISO format")
    appointment_type: str = Field(..., description="Type of appointment (e.g., general physician, specialist)")

class ModifyAppointmentInput(BaseModel):
    subscription_id: str = Field(..., description="Subscription ID associated with the appointment")
    new_appointment_date: str = Field(..., description="New scheduled date of the appointment in ISO format")
    new_appointment_type: str = Field(..., description="New type of appointment")

class CancelAppointmentInput(BaseModel):
    subscription_id: str = Field(..., description="Subscription ID associated with the appointment")

# Validation for appointment date
def validate_appointment_date(appointment_date):
    try:
        appointment_datetime = datetime.fromisoformat(appointment_date)
        now = DateTimeManager.now()
        if not (now + timedelta(hours=24) <= appointment_datetime <= now + timedelta(days=30)):
            return False, "Appointment date must be more than 24 hours in the future and within the next 30 days."
        if appointment_datetime.minute not in [0, 30]:
            return False, "Appointments can only be booked at full or half-hour intervals."
        return True, ""
    except ValueError as e:
        return False, f"Invalid date format: {str(e)}"

def validate_cancellation_date(appointment_date):
    try:
        appointment_datetime = datetime.fromisoformat(appointment_date)
        now = DateTimeManager.now()
        if appointment_datetime - timedelta(hours=24) <= now:
            return False, "Appointments can only be cancelled at least 24 hours in advance."
        return True, ""
    except ValueError as e:
        return False, f"Invalid date format: {str(e)}"

def check_active_appointments(subscription_id, db):
    try:
        query = f"""
        SELECT appointment_date FROM customer_appointments
        WHERE subscription_id = '{subscription_id}' AND appointment_date >= '{DateTimeManager.iso_now()}'
        """
        result = db.run(query)
        if result:
            return False, "User already has an active appointment and cannot book another."
        return True, ""
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def check_appointments(subscription_id, db):
    """Retrieve appointments from the database."""
    try:
        query = f"""
        SELECT * FROM customer_appointments
        WHERE subscription_id = '{subscription_id}'
        """
        result = db.run(query)

        # Handle no results found
        if not result:
            return "No appointments found.", {}

        metadata = {"subscription_id": subscription_id, "query": query}
        return result, metadata

    except Exception as e:
        return f"An error occurred: {str(e)}", {}

def create_appointment(subscription_id, appointment_date, appointment_type, db):
    """Create a new appointment."""
    try:
        is_valid, message = validate_appointment_date(appointment_date)
        if not is_valid:
            return {"status": "error", "message": message}

        is_valid, message = check_active_appointments(subscription_id, db)
        if not is_valid:
            return {"status": "error", "message": message}

        query = f"""
        INSERT INTO customer_appointments (subscription_id, appointment_created_date, appointment_date, appointment_type)
        VALUES ('{subscription_id}', '{DateTimeManager.iso_now()}', '{appointment_date}', '{appointment_type}')
        """
        db.run(query)
        metadata = {"subscription_id": subscription_id, "appointment_date": appointment_date, "appointment_type": appointment_type}
        return {"status": "success", "message": "Appointment created successfully.", "metadata": metadata}

    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

def modify_appointment(subscription_id, new_appointment_date, new_appointment_type, db):
    """Modify an existing appointment."""
    try:
        is_valid, message = validate_appointment_date(new_appointment_date)
        if not is_valid:
            return {"status": "error", "message": message}

        query = f"""
        UPDATE customer_appointments
        SET appointment_date = '{new_appointment_date}', appointment_type = '{new_appointment_type}'
        WHERE subscription_id = {subscription_id} and appointment_date >= '{DateTimeManager.iso_now()}'
        """
        db.run(query)
        metadata = {"subscription_id": subscription_id, "new_appointment_date": new_appointment_date, "new_appointment_type": new_appointment_type}
        return {"status": "success", "message": "Appointment modified successfully.", "metadata": metadata}

    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

def cancel_appointment(subsription_id, db):
    """Cancel an existing appointment."""
    try:
        # Fetch appointment details
        query_fetch = f"""
        SELECT appointment_date FROM customer_appointments WHERE subscription_id = '{subsription_id}' and appointment_date >= '{DateTimeManager.iso_now()}'
        """
        result = db.run(query_fetch)
        if not result:
            return {"status": "error", "message": f"No appointment found with subscription_id {subsription_id}."}

        appointment_date = result[0]["appointment_date"]
        is_valid, message = validate_cancellation_date(appointment_date)
        if not is_valid:
            return {"status": "error", "message": message}

        # Perform cancellation
        query_cancel = f"""
        DELETE FROM customer_appointments WHERE subscription_id = '{subsription_id}' and appointment_date >= '{DateTimeManager.iso_now()}'
        """
        db.run(query_cancel)
        metadata = {"subsription_id": subsription_id}
        return {"status": "success", "message": "Appointment cancelled successfully.", "metadata": metadata}

    except Exception as e:
        return {"status": "error", "message": f"An error occurred: {str(e)}"}

###############################################################################
# Agent Tools
###############################################################################

def create_check_appointments_tool(db):
    return StructuredTool.from_function(
        func=lambda subscription_id: check_appointments(subscription_id, db),
        name="CheckAppointmentsTool",
        description="Retrieve all appointments for a given subscription ID.",
        args_schema=CheckAppointmentsInput,
    )

def create_create_appointment_tool(db):
    return StructuredTool.from_function(
        func=lambda subscription_id, appointment_date, appointment_type: create_appointment(
            subscription_id, appointment_date, appointment_type, db
        ),
        name="CreateAppointmentTool",
        description="Create a new appointment for a subscription ID.",
        args_schema=AppointmentInput,
    )

def create_modify_appointment_tool(db):
    return StructuredTool.from_function(
        func=lambda subscription_id, new_appointment_date, new_appointment_type: modify_appointment(
            subscription_id, new_appointment_date, new_appointment_type, db
        ),
        name="ModifyAppointmentTool",
        description="Modify an existing appointment by subscription_id.",
        args_schema=ModifyAppointmentInput,
    )

def create_cancel_appointment_tool(db):
    return StructuredTool.from_function(
        func=lambda subscription_id: cancel_appointment(subscription_id, db),
        name="CancelAppointmentTool",
        description="Cancel an appointment by subscription_id. Appointments can only be cancelled at least 24 hours in advance.",
        args_schema=CancelAppointmentInput,
    )

###############################################################################
# Agent Creation
###############################################################################

def create_appointment_agent(db=db, llm=llm, checkpointer=MemorySaver()):
    # Prompt
    system_message = f"""
    You are an agent managing customer appointments in a SQL database. The current system date is {DateTimeManager.now().isoformat()}.
    You can check, create, modify, or cancel appointments. Ensure that new or modified appointments.
    Only interact with the database through the provided tools.
    """

    # Tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    toolkit += [
        create_check_appointments_tool(db),
        create_create_appointment_tool(db),
        create_modify_appointment_tool(db),
        create_cancel_appointment_tool(db),
    ]

    # Create agent
    agent = create_react_agent(
        llm,
        tools=toolkit,
        state_modifier=system_message,
        checkpointer=checkpointer,
    )

    return agent


""" # Run a sample query
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
appointment_agent = create_appointment_agent()

thread_id = 1
config = {"configurable": {"thread_id": thread_id}}
state = MessagesState(
    messages=[HumanMessage(content=f"Create an appointment for subscription_id SUB10011")],
    thread_id=thread_id,
    customer_id=customer_id
)

response = appointment_agent.invoke(state, config=config)
response["messages"][-1].pretty_print()

state = MessagesState(
    messages=[HumanMessage(content=f"cancel my appointment")],
    thread_id=thread_id,
    customer_id=customer_id
)
"""
