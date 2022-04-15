import pandas as pd
import click
import os
import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://labeler.oz-services.ru/"


def get_file(path: str, destination: Path):
    server_path = Path(path)
    response = requests.get(BASE_URL + path)
    if not os.path.exists(destination / server_path.parent.parent.name):
        os.makedirs(destination / server_path.parent.parent.name)
    if not os.path.exists(destination / server_path.parent.parent.name / server_path.parent.name):
        os.makedirs(destination / server_path.parent.parent.name / server_path.parent.name)
    with (destination / server_path.parent.parent.name / server_path.parent.name / server_path.name).open("wb") as file:
        file.write(response.content)


def load_files(csv_filename: str, destination: Path):
    df = pd.read_csv(csv_filename, index_col=0, sep='\t')
    for frame_path in tqdm(df.frame_filename, total=len(df.path), desc="Load files"):
        get_file(frame_path, destination)


@click.command()
@click.option('--csv_filename', default='/home/yandex/igor/data_eyes/frames.eyes-eubank.csv', type=str)
@click.option('--destination', default='/home/yandex/igor/data_eyes/eubank', type=str)
def fmain(csv_filename, destination):
    destination = Path(destination)
    if Path(csv_filename).exists():
        load_files(csv_filename, destination)
    else:
        exit(1)


if __name__ == "__main__":
    fmain()
