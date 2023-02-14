import requests
import shutil
import torch
from ..config import SPAI_URL, retrieve_credentials


def upload(model_path, name):
    creds = retrieve_credentials()
    # files = {'file': open(model_path, 'rb'), 'name': name}
    files = {"file": open(model_path, "rb")}
    r = requests.post(
        f"{SPAI_URL}/models",
        data={"name": name},
        files=files,
        headers={"Authorization": "Bearer " + creds["id_token"]},
    )
    return r.json()


def download(model_name, dest_path=".", dest_name=None):
    dest_name = dest_name if dest_name is not None else model_name
    dest_path = f"{dest_path}/{dest_name}"
    creds = retrieve_credentials()
    try:
        model = torch.load(dest_path)
        print("model found")
        return model
    except:
        print("model not found, downloading ...", end=" ")
        url = f"{SPAI_URL}/models/{model_name}"
        response = requests.get(
            url, stream=True, headers={"Authorization": "Bearer " + creds["id_token"]}
        )
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        print("done")
        return torch.jit.load(dest_path)
