import os

import anndata as ad
import boto3
import botocore
import numpy as np
import pandas as pd

SPECTRA_DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "spectra_cache")


def read_aws_h5ad(s3_url):
    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    s3_object.download_file(save_path)

    adata = ad.read_h5ad(save_path)
    return adata


def read_aws_csv(s3_url, sep=","):
    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    try:
        s3_object.download_file(save_path)
        df = pd.read_csv(save_path, sep=sep)
        return df
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("object does not exist")
            return None
    return None


def read_aws_npz(s3_url, sep=","):
    save_path = os.path.join(SPECTRA_DEFAULT_DIR, s3_url.split("/")[-1])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s3 = boto3.resource("s3")

    # Get the bucket name and key from the s3 url
    bucket_name, key = s3_url.removeprefix("s3://").split("/", 1)

    s3_object = s3.Object(bucket_name=bucket_name, key=key)
    try:
        s3_object.download_file(save_path)
        mtx = np.laod(save_path)
        return mtx
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("object does not exist")
            return None
    return None
