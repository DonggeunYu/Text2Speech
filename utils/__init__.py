import os
import re
import json
from collections import namedtuple
from datetime import datetime, timedelta
from shutil import copyfile as copy_file
from contextlib import closing
from multiprocessing import Pool
from tqdm import tqdm

PARAMS_NAME = "params.json"

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []

def prepare_dirs(config, hparams):
    if hasattr(config, "data_paths"):
        config.datasets = [os.path.basename(data_path) for data_path in config.data_paths]
        dataset_desc = "+".join(config.datasets)

    if config.load_path:
        config.model_dir = config.load_path
    else:
        config.model_name = "{}_{}".format(dataset_desc, get_time())
        config.model_dir = os.path.join(config.log_dir, config.model_name)

        for path in [config.log_dir, config.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    if config.load_path:
        load_hparams(hparams, config.model_dir)
    else:
        hparams["num_speakers"] = len(config.datasets)

        save_hparams(config.model_dir, hparams)
        copy_file("hparams.py", os.path.join(config.model_dir, "hparams.py"))

def parallel_run(fn, items, desc="", parallel=True):
    results = []

    if parallel:
        with closing(Pool()) as pool:
            for out in tqdm(pool.imap_unordered(fn, items), total=len(items), desc=desc):
                if out is not None:
                    results.append(out)
    else:
        for item in tqdm(items, total=len(items), desc=desc):
            out = fn(item)
            if out is not None:
                results.append(out)

    return results


def save_hparams(model_dir, hparams):
    param_path = os.path.join(model_dir, PARAMS_NAME)

    info = hparams
    write_json(param_path, info)

    print(" [*] MODEL dir: {}".format(model_dir))
    print(" [*] PARAM path: {}".format(param_path))


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)


def load_json(path, as_class=False, encoding='euc-kr'):
    with open(path, encoding=encoding) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook= \
                lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)


def load_hparams(hparams, load_path, skip_list=[]):
    # log dir에 있는 hypermarameter 정보를 이용해서, hparams.py의 정보를 update한다.
    path = os.path.join(load_path, PARAMS_NAME)

    new_hparams = load_json(path)
    hparams_keys = vars(hparams).keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))  # json에 있지만, hparams에 없다는 의미
            continue

        if key not in ['xxxxx', ]:  # update 하지 말아야 할 것을 지정할 수 있다.
            original_value = getattr(hparams, key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, getattr(hparams, key), value))
                setattr(hparams, key, value)

def remove_file(path):
    if os.path.exists(path):
        print(" [*] Removed: {}".format(path))
        os.remove(path)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_H-%M-%S")
