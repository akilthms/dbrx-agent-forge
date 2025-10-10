from databricks.sdk import WorkspaceClient
import mlflow
from pydantic_settings import BaseSettings

w = WorkspaceClient()

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("")
mlflow.autolog()
catalog = w.catalogs.get(name="akthom")

def
