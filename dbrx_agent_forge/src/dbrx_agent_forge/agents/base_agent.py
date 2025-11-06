from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.tools import Tool
from typing import List
from mlflow.entities import Experiment

from dbrx_agent_forge.src.dbrx_agent_forge.utils.mlflow_utils import get_or_create_mlflow_experiment



class BaseAgent(ABC):
    """Abstract base class for agent creation, deployment and evaluation."""

    def __init__(self, experiment_name: str, artifact_location: Optional[str] = None):
        """Initialize the base agent with MLflow experiment tracking.

        Args:
            experiment_name: Name of the MLflow experiment for tracking
            artifact_location: Optional artifact location for MLflow tracking
        """
        self.experiment = get_or_create_mlflow_experiment(
            experiment_name=experiment_name,
            artifact_location=artifact_location
        )

    @abstractmethod
    def get_tools(self, tools:List[Tool]) -> List[Tool]:
        """Retrieve a list of tools available to the agent.

        Args:
            tools: List of tool instances to be used by the agent

        Returns:
            List of tool instances
        """
        raise NotImplementedError

    @abstractmethod
    def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create a new agent instance.

        Args:
            config: Configuration parameters for agent creation

        Returns:
            Created agent instance
        """
        raise NotImplementedError

    @abstractmethod
    def deploy_agent(self, agent: Any) -> bool:
        """Deploy the agent to the target environment.

        Args:
            agent: Agent instance to deploy

        Returns:
            True if deployment was successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_agent(self, agent: Any) -> Dict[str, Any]:
        """Evaluate the agent's performance.

        Args:
            agent: Agent instance to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        raise NotImplementedError
