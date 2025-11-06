import mlflow
from dbrx_agent_forge.src.dbrx_agent_forge.utils.mlflow_utils import get_or_create_mlflow_experiment
from abc import ABC, abstractmethod

CATALOG = ""
SCHEMA = ""
PROMPT_REGISTRY = f"{CATALOG}.{SCHEMA}.agent_forge_prompt_registry"


class PromptRegistry(ABC):
    """Manage Prompt Registry via MLFlow"""
    def __init__(self, prompt_registry:str, experiment_name:str):
        self.experiment = get_or_create_mlflow_experiment(
            experiment_name=experiment_name,
        )
        self.prompt_registry = prompt_registry

    @staticmethod
    def register_prompt(self, template, commit_message: str ,catalog: str = None, schema: str=None):
        # Register a prompt template
        prompt = mlflow.genai.register_prompt(
            name=",
            template="You are a helpful assistant. Answer this question: {{question}}",
            commit_message="Initial customer support prompt"
        )
        print(f"Created version {prompt.version}")


    @staticmethod
    def load_prompt(self, name:str=None, uri:str=None):
        return mlflow.genai.load_prompt(name_or_uri=name or uri)
