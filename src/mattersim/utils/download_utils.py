# -*- coding: utf-8 -*-
"""
This module contains utility functions for downloading files.
"""
import os

import requests
from loguru import logger


def download_file(url: str, output_path: str):
    """
    A wrapper around requests.get to download a file from a URL.

    Args:
        url (str): The URL to download the file from.
        output_path (str): The path to save the downloaded file to.
    """

    logger.info(f"Downloading file from {url} to {output_path}")
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)
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
