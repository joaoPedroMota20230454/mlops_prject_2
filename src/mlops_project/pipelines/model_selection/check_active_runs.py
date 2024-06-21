from mlflow.tracking import MlflowClient
import mlflow

def get_active_runs(experiment_id):
    """
    Get a list of active runs for a given experiment.

    Args:
        experiment_id (str): The ID of the experiment to check for active runs.

    Returns:
        List of active run IDs.
    """
    client = MlflowClient()
    # Search for active runs
    active_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'RUNNING'"
    )
    active_run_ids = [run.info.run_id for run in active_runs]
    return active_run_ids

def end_active_runs(experiment_id):
    """
    End all active runs for a given experiment.

    Args:
        experiment_id (str): The ID of the experiment to end active runs.
    """
    client = MlflowClient()
    active_run_ids = get_active_runs(experiment_id)
    for run_id in active_run_ids:
        client.set_terminated(run_id)
        print(f"Ended run with ID: {run_id}")

# Example usage
if __name__ == "__main__":
    experiment_id = "0"  # Replace with your experiment ID
    active_runs = get_active_runs(experiment_id)

    if active_runs:
        print(f"Active runs found: {active_runs}")
        # Optionally, end all active runs
        end_active_runs(experiment_id)
    else:
        print("No active runs found.")
