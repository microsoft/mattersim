"""
This module contains utility functions for downloading files.
"""

import os

import requests
from loguru import logger
from tqdm import tqdm


def download_file(url: str, output_path: str):
    """
    A wrapper around requests.get to download a file from a URL,
    with a progress bar for large files.

    Args:
        url (str): The URL to download the file from.
        output_path (str): The path to save the downloaded file to.
    """

    logger.info(f"Downloading file from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with (
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(output_path),
            disable=total_size == 0,
        ) as progress_bar,
        open(output_path, "wb") as f,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            progress_bar.update(len(chunk))

    logger.info(f"File downloaded to {output_path}")


def download_checkpoint(
    checkpoint_name: str, save_folder: str = "~/.local/mattersim/pretrained_models/"
):
    """
    Download a checkpoint from the Microsoft Mattersim repository.

    Args:
        checkpoint_name (str): The name of the checkpoint to download.
        save_folder (str): The local folder to save the checkpoint to.
    """

    GITHUB_CHECKPOINT_PREFIX = (
        "https://raw.githubusercontent.com/microsoft/mattersim/main/pretrained_models/"
    )
    checkpoint_url = GITHUB_CHECKPOINT_PREFIX + checkpoint_name.strip("/")
    save_path = os.path.join(
        os.path.expanduser(save_folder), checkpoint_name.strip("/")
    )
    download_file(checkpoint_url, save_path)
