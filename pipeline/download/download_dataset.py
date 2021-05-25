import urllib.request
import tarfile
import os

def download_dataset():
    """Download the UIT-VSFC dataset to the KFP volume to share it among all steps"""

    print("Start Downloading...")
    url = "http://minio.kubeflow:9000/sentiment-analysis/datasets.tar.gz"
    # url = "http://localhost:9000/sentiment-analysis/datasets.tar.gz"
    stream = urllib.request.urlopen(url)
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    tar.extractall("/app")
    print("Download and Extract Successfully")

if __name__ == "__main__":
    download_dataset()
