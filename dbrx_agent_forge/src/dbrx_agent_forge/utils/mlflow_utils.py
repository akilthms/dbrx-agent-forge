
from typing import Optional
import mlflow
from mlflow.entities import Experiment


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