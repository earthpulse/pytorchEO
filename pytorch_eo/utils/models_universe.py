import requests
import shutil
import torch
from ..config import MU_API_URL


def upload(model_path, name=None):
    file = open(model_path, 'rb')
    files = {'file': file}
    name = name if name is not None else file.filename
    r = requests.post(f'{MU_API_URL}/upload/{name}', files=files)
    return r.json()


def download(model_name, dest_path='.', dest_name=None):
    dest_name = dest_name if dest_name is not None else model_name
    dest_path = f'{dest_path}/{dest_name}'
    try:
        model = torch.load(dest_path)
        print('model found')
        return model
    except:
        print('model not found, downloading ...', end=" ")
        url = f"{MU_API_URL}/download/{model_name}"
        response = requests.get(url, stream=True)
        with open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        print("done")
        return torch.jit.load(dest_path)
