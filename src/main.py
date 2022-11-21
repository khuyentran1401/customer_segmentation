from process_data import process_data
from segment import segment
from run_notebook import run_notebook
from prefect import flow


@flow
def main():
    process_data()
    segment()
    run_notebook()


if __name__ == "__main__":
    main()
