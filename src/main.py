from process_data import process_data
from segment import segment
from prefect import flow

@flow
def main():
    process_data()
    segment()


if __name__ == "__main__":
    main()
