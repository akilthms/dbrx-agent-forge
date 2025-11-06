from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (TableInfo,
                                            ColumnInfo,
                                            DataSourceFormat,
                                            ColumnTypeName,
                                            TableType)
import mlflow
import mlflow
from typing import Union, Optional
from mlflow.entities import Experiment
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain_core.runnables import Runnable
from pydantic_settings import BaseSettings
from dbrx_agent_forge.src.dbrx_agent_forge.config import Settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from databricks_langchain import ChatDatabricks
from langchain.agents import create_tool_calling_agent, AgentType, AgentExecutor
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated, List
from langgraph.graph.message import AnyMessage, add_messages, MessagesState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from dbrx_agent_forge.src.dbrx_agent_forge.agents.base_agent import BaseAgent


class UCAgent(BaseAgent):
    def __init__(self, experiment_name: str, settings: Settings = Settings()):
        super().__init__(experiment_name)
        self.settings = settings


    @tool
    def read_table(
            self,
            catalog_name: str,
            schema_name: str,
            table_name: str,
            limit: Optional[int] = 1000,
            where_clause: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Read contents of a Unity Catalog table.

        Args:
            catalog_name: Name of the catalog
            schema_name: Name of the schema
            table_name: Name of the table
            limit: Maximum number of rows to return (default: 1000)
            where_clause: Optional WHERE clause for filtering

        Returns:
            List of dictionaries representing table rows
        """
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

        # Build SELECT statement
        sql = f"SELECT * FROM {full_table_name}"

        if where_clause:
            sql += f" WHERE {where_clause}"

        if limit:
            sql += f" LIMIT {limit}"

        try:
            result = w.statement_execution.execute_statement(
                warehouse_id=settings.warehouse_id,
                statement=sql
            )

            # Wait for completion and get results
            if result.result and result.result.data_array:
                # Get column names from table schema
                table_info = w.tables.get(full_table_name)
                column_names = [col.name for col in table_info.columns]

                # Convert result data to list of dictionaries
                rows = []
                for row_data in result.result.data_array:
                    row_dict = {}
                    for i, value in enumerate(row_data):
                        if i < len(column_names):
                            row_dict[column_names[i]] = value
                    rows.append(row_dict)

                return rows
            else:
                return []

        except Exception as e:
            return [{"error": f"Error reading table: {str(e)}"}]


class AgentState(TypedDict):
    # This key holds the list of messages (history)
    messages: Annotated[List[AnyMessage], add_messages]
    # This key tracks the last tool call/response
    tool_calls: List[dict]



def get_or_create_mlflow_experiment(
            experiment_name: str,
            artifact_location: Optional[str] = None
    ) -> Experiment:
        """
        Retrieves an existing MLflow Experiment by name or creates a new one.

        Args:
            experiment_name: The name of the MLflow experiment.
            artifact_location: The artifact URI where experiment artifacts should be stored.
                               (Only used if the experiment is created).

        Returns:
            The MLflow Experiment object.
        """

        # 1. Try to retrieve the experiment by name
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment:
            print(f"✅ Found existing experiment '{experiment_name}' (ID: {experiment.experiment_id}).")
            return experiment
        else:
            # 2. Create the experiment if it does not exist
            try:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
                experiment = mlflow.get_experiment(experiment_id)
                print(f"✨ Created new experiment '{experiment_name}' (ID: {experiment_id}).")
                return experiment
            except Exception as e:
                print(f"❌ Error creating experiment '{experiment_name}': {e}")
                raise


def _get_type_json(column_type: str) -> str:
    """
    Convert column type to JSON format required by Unity Catalog.

    Args:
        column_type: The column type as string (e.g., 'INT', 'STRING', 'DOUBLE')

    Returns:
        JSON representation of the type
    """
    type_upper = column_type.upper()

    # Map common types to their JSON representations
    type_mapping = {
        'INT': '"integer"',
        'INTEGER': '"integer"',
        'LONG': '"long"',
        'BIGINT': '"long"',
        'STRING': '"string"',
        'VARCHAR': '"string"',
        'TEXT': '"string"',
        'DOUBLE': '"double"',
        'FLOAT': '"float"',
        'BOOLEAN': '"boolean"',
        'BOOL': '"boolean"',
        'DATE': '"date"',
        'TIMESTAMP': '"timestamp"',
        'TIMESTAMP_NTZ': '"timestamp_ntz"',
        'BINARY': '"binary"',
        'DECIMAL': '"decimal(10,0)"',  # Default precision and scale
        'ARRAY': '"array<string>"',  # Default array type
        'MAP': '"map<string,string>"',  # Default map type
        'STRUCT': '"struct<>"'  # Empty struct
    }

    return type_mapping.get(type_upper, f'"{type_upper.lower()}"')


@tool
def create_table(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    columns: List[Dict[str, str]],
    data_source_format: str = "DELTA",
    storage_location: Optional[str] = Settings().external_storage_location
) -> str:
    """
    Create a new table in Unity Catalog.

    Args:
        catalog_name: Name of the catalog
        schema_name: Name of the schema
        table_name: Name of the table
        columns: List of column definitions with 'name' and 'type' keys. Supported data types include:    Supported data types include:
                 ARRAY, BINARY, BOOLEAN, BYTE, CHAR, DATE, DECIMAL, DOUBLE, FLOAT,
                 GEOGRAPHY, GEOMETRY, INT, INTERVAL, LONG, MAP, NULL, SHORT,
                 STRING, STRUCT, TABLE_TYPE, TIMESTAMP, TIMESTAMP_NTZ,
                 USER_DEFINED_TYPE, VARIANT
        data_source_format: Format of the table (default: DELTA)
        storage_location: Optional storage location for external tables

    Returns:
        Full table name that was created
    """
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

    column_infos = [
        ColumnInfo(
            name=col["name"],
            type_text=col["type"],
            type_name=ColumnTypeName(col["type"].upper()),
            type_json=_get_type_json(col["type"]),
            position=i  # Set unique position for each column
        ) for i, col in enumerate(columns)
    ]


    table_info = TableInfo(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=TableType("MANAGED") if storage_location is None else TableType("EXTERNAL"),
        data_source_format=DataSourceFormat(data_source_format.upper()),
        columns=column_infos,
        storage_location=storage_location
    )

    w.tables.create(**table_info.as_shallow_dict())
    return full_table_name

@tool
def append_rows(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    data: List[Dict[str, Any]]
) -> str:
    """
    Append rows to an existing Unity Catalog table using SQL INSERT.

    Args:
        catalog_name: Name of the catalog
        schema_name: Name of the schema
        table_name: Name of the table
        data: List of dictionaries representing rows to insert

    Returns:
        Success message
    """
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

    if not data:
        return "No data to insert"

    # Get table schema to ensure proper column ordering
    table_info = w.tables.get(full_table_name)
    column_names = [col.name for col in table_info.columns]

    # Build INSERT statement
    columns_str = ", ".join(column_names)

    values_list = []
    for row in data:
        values = []
        for col in column_names:
            value = row.get(col)
            if value is None:
                values.append("NULL")
            elif isinstance(value, str):
                values.append(f"'{value.replace("'", "''")}'")
            else:
                values.append(str(value))
        values_list.append(f"({', '.join(values)})")

    values_str = ", ".join(values_list)
    sql = f"INSERT INTO {full_table_name} ({columns_str}) VALUES {values_str}"

    w.statement_execution.execute_statement(
        warehouse_id=settings.warehouse_id,
        statement=sql
    )

    return f"Successfully inserted {len(data)} rows into {full_table_name}"

@tool
def read_volume(
    catalog_name: str,
    schema_name: str,
    volume_name: str,
    file_path: Optional[str] = None
) -> List[str]:
    """
    Read contents from a Unity Catalog volume.

    Args:
        catalog_name: Name of the catalog
        schema_name: Name of the schema
        volume_name: Name of the volume
        file_path: Optional specific file path within the volume

    Returns:
        List of file paths or file contents
    """
    volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

    if file_path:
        full_path = f"{volume_path}/{file_path.lstrip('/')}"
        try:
            # Use workspace client to read file
            with w.workspace.download(full_path) as file_content:
                return [file_content.read().decode('utf-8')]
        except Exception as e:
            return [f"Error reading file: {str(e)}"]
    else:
        # List files in volume
        try:
            files = w.files.list(volume_path)
            return [file.path for file in files]
        except Exception as e:
            return [f"Error listing volume contents: {str(e)}"]

@tool
def merge_data(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    source_data: List[Dict[str, Any]],
    merge_keys: List[str],
    update_columns: Optional[List[str]] = None
) -> str:
    """
    Merge data into an existing Unity Catalog table using MERGE statement.

    Args:
        catalog_name: Name of the catalog
        schema_name: Name of the schema
        table_name: Name of the table
        source_data: List of dictionaries representing source data
        merge_keys: List of column names to use for matching
        update_columns: Optional list of columns to update (defaults to all non-key columns)

    Returns:
        Success message
    """
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

    if not source_data:
        return "No data to merge"

    # Get table schema
    table_info = w.tables.get(full_table_name)
    all_columns = [col.name for col in table_info.columns]

    # Determine update columns
    if update_columns is None:
        update_columns = [col for col in all_columns if col not in merge_keys]

    # Create temporary view from source data
    temp_view_name = f"temp_merge_source_{table_name}"

    # Build VALUES clause for source data
    values_list = []
    for row in source_data:
        values = []
        for col in all_columns:
            value = row.get(col)
            if value is None:
                values.append("NULL")
            elif isinstance(value, str):
                values.append(f"'{value.replace("'", "''")}'")
            else:
                values.append(str(value))
        values_list.append(f"({', '.join(values)})")

    columns_str = ", ".join(all_columns)
    values_str = ", ".join(values_list)

    # Create temporary view
    create_view_sql = f"""
    CREATE OR REPLACE TEMPORARY VIEW {temp_view_name} AS
    SELECT * FROM VALUES {values_str} AS t({columns_str})
    """

    # Build MERGE statement
    merge_conditions = " AND ".join([f"target.{key} = source.{key}" for key in merge_keys])
    update_assignments = ", ".join([f"{col} = source.{col}" for col in update_columns])
    insert_columns = ", ".join(all_columns)
    insert_values = ", ".join([f"source.{col}" for col in all_columns])

    merge_sql = f"""
    MERGE INTO {full_table_name} AS target
    USING {temp_view_name} AS source
    ON {merge_conditions}
    WHEN MATCHED THEN
        UPDATE SET {update_assignments}
    WHEN NOT MATCHED THEN
        INSERT ({insert_columns}) VALUES ({insert_values})
    """

    try:
        # Execute both statements
        w.statement_execution.execute_statement(
            warehouse_id=settings.warehouse_id,
            statement=create_view_sql
        )

        w.statement_execution.execute_statement(
            warehouse_id=settings.warehouse_id,
            statement=merge_sql
        )

        return f"Successfully merged {len(source_data)} rows into {full_table_name}"
    except Exception as e:
        return f"Error during merge operation: {str(e)}"

@tool
def read_table(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    limit: Optional[int] = 1000,
    where_clause: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Read contents of a Unity Catalog table.

    Args:
        catalog_name: Name of the catalog
        schema_name: Name of the schema
        table_name: Name of the table
        limit: Maximum number of rows to return (default: 1000)
        where_clause: Optional WHERE clause for filtering

    Returns:
        List of dictionaries representing table rows
    """
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

    # Build SELECT statement
    sql = f"SELECT * FROM {full_table_name}"

    if where_clause:
        sql += f" WHERE {where_clause}"

    if limit:
        sql += f" LIMIT {limit}"

    try:
        result = w.statement_execution.execute_statement(
            warehouse_id=settings.warehouse_id,
            statement=sql
        )

        # Wait for completion and get results
        if result.result and result.result.data_array:
            # Get column names from table schema
            table_info = w.tables.get(full_table_name)
            column_names = [col.name for col in table_info.columns]

            # Convert result data to list of dictionaries
            rows = []
            for row_data in result.result.data_array:
                row_dict = {}
                for i, value in enumerate(row_data):
                    if i < len(column_names):
                        row_dict[column_names[i]] = value
                rows.append(row_dict)

            return rows
        else:
            return []

    except Exception as e:
        return [{"error": f"Error reading table: {str(e)}"}]


# tools = [create_table, append_rows, read_volume, read_table, merge_data]


def create_langchain_agent(tools: List[BaseTool]=None) -> Runnable:
    llm = ChatDatabricks(
        model="databricks-claude-sonnet-4",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized agent that can create,
        read, append to, and merge Unity Catalog tables. You have access to
        tools to perform these operations. When a user asks you to create a table,
        use the create_table tool with the appropriate parameters."""),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def create_langgraph_agent(tools: List[BaseTool]=None) -> Runnable:
    """Create a LangGraph agent with Unity Catalog tools."""

    llm = ChatDatabricks(
        model="databricks-claude-sonnet-4",
        temperature=0.1
    )

    llm_with_tools = llm.bind_tools(tools)

    SYSTEM_PROMPT = """
        You are a specialized agent that can create,
        read, append to, and merge Unity Catalog tables. You have access to
        tools to perform these operations. When a user asks you to create a table,
        use the create_table tool with the appropriate parameters."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    def get_latest_message(messages: List[AnyMessage]) -> AnyMessage:
        """Extract the latest message from a list of messages."""
        return next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

    def agent_node(state:  MessagesState)-> AIMessage:
        last_message = get_latest_message(state["messages"])
        messages = prompt.format_messages(input=last_message)
        return AIMessage(content=llm_with_tools.invoke(messages).content)

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)
    # ➕ Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    # ➕ Entry point and Edges
    workflow.set_entry_point("agent")
    workflow.add_edge("tools", "agent")
    workflow.add_condtional_edge("agent",
                                 tools_condition,
                                 {
                                     "tools": "tools",
                                     "__end__": END
                                 }
                                 )

    agent = workflow.complile()
    print("❔Type of agent:", type(agent))
    return agent





def init_mlflow_env(settings:Settings):
    mlflow.set_tracking_uri("databricks")
    get_or_create_mlflow_experiment(settings.experiment_path)
    mlflow.set_experiment(settings.experiment_path)
    mlflow.autolog()

def test_create_table(agent: Optional[Runnable] = None):
    """Test creating a Unity Catalog table using the agent."""
    response = agent.invoke({"input": """
        Please create a table called 'test_table' in the 'akthom' 
        catalog and 'agents' schema with the following columns:
        
        - id (integer)
        - name (string)
        - age (integer)
        - email (string)
    """})

    print(response["output"])

def test_append_rows(agent: Optional[Runnable] = None):
    """Test appending rows to a Unity Catalog table using the agent."""
    response = agent.invoke(
        {
            "input": """
                    Please append the following rows to the 'test_table' in the 'akthom' 
                    catalog and 'agents' schema:
                    - (1, 'Alice', 30, 'alice@databricks.com')
                    """
        }
    )

    print(response["output"])


if __name__ == "__main__":
    w = WorkspaceClient()
    init_mlflow_env(Settings())
    tools = [create_table, append_rows]#, read_volume, read_table, merge_data]
    #agent = create_langchain_agent()
    agent = create_langgraph_agent(tools)
    #test_create_table(agent)
    test_append_rows(agent)