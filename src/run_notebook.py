from prefect import flow
from prefect_hex import HexCredentials

from prefect_hex.project import get_project_runs
from prefect_hex import HexCredentials
from prefect.blocks.system import Secret


@flow
def get_project_runs_flow():
    hex_credentials = HexCredentials.load("article-demo")
    project_id = Secret.load("hex-project-id").get()
    return get_project_runs(
        project_id=project_id,
        hex_credentials=hex_credentials,
    )


if __name__ == "__main__":
    get_project_runs_flow()
