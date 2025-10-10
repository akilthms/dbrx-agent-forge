from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import TableInfo, ColumnInfo, DataSourceFormat
import mlflow
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain_core.runnables import Runnable
from pydantic_settings import BaseSettings
from dbrx_agent_forge.src.dbrx_agent_forge.config import Settings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from databricks_langchain import ChatDatabricks
from langchain.agents import create_tool_calling_agent, AgentType, AgentExecutor
from langgraph.prebuilt import create_react_agent


w = WorkspaceClient()
settings = Settings()
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(settings.experiment_path)
mlflow.autolog()
catalog = w.catalogs.get(name="akthom")

@tool
def create_table(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    columns: List[Dict[str, str]],
    data_source_format: str = "DELTA",
    storage_location: Optional[str] = None
) -> str:
    """
    Create a new table in Unity Catalog.

    Args:
        catalog_name: Name of the catalog
        schema_name: Name of the schema
        table_name: Name of the table
        columns: List of column definitions with 'name' and 'type' keys
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
            type_name=col["type"]
        ) for col in columns
    ]

    table_info = TableInfo(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type="MANAGED" if storage_location is None else "EXTERNAL",
        data_source_format=DataSourceFormat(data_source_format.upper()),
        columns=column_infos,
        storage_location=storage_location
    )

    w.tables.create(table_info)
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


tools = [create_table, append_rows, read_volume, read_table, merge_data]


def create_langchain_agent() -> Runnable:
    llm = ChatDatabricks(
        model="databricks/databricks-claude-sonnet-4",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template("""
       You are a specialized agent that only has the ability to create, read, append to, 
       and merge a unity catalog table.
    """)

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor

def create_langgraph_agent():
    llm = ChatDatabricks(
        model="databricks/databricks-claude-sonnet-4",
        temperature=0.1
    )

    app = create_react_agent(llm, tools)

def test_create_table():
    agent = create_langchain_agent()
