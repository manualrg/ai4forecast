import os
import yaml
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

root_path = os.environ.get("LOCAL_PATH")
data_path = os.path.join(root_path, "data", "")

raw_path = os.path.join(data_path, "raw", "")
interim_path = os.path.join(data_path, "interim", "")
processed_path = os.path.join(data_path, "processed", "")

models_path = os.path.join(root_path, "models", "")

def get_conf(path, filename='conf.yaml'):
    filename_conf = os.path.join(path, filename)

    with open(filename_conf, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)

    return cfg


conf = get_conf(path=root_path)


def get_paths():


    return {'raw': raw_path,
            'interim': interim_path,
            'processed': processed_path,
            'models': models_path
            }