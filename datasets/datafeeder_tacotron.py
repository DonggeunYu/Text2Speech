from collections import defaultdict
from utils.infolog import log
from utils import parallel_run, remove_file
from utils.audio import frames_to_hours

from glob import glob
import pprint
import os
import time
import torch
import numpy as np

from torch.utils import data as data_utils

_pad = 0


def get_frame(path):
    try:
        data = np.load(path)
        n_frame = data["linear"].shape[0]
        n_token = len(data["tokens"])
        return (path, n_frame, n_token)
    except:
        print(path)


def get_path_dict(data_dirs, hparams, config, data_type, n_test=None, rng=np.random.RandomState(123)):
    # Load metadata:
    path_dict = {}
    for data_dir in data_dirs:  # ['datasets/moon\\data']
        paths = glob("{}/*.npz".format(
            data_dir))  # ['datasets/moon\\data\\001.0000.npz', 'datasets/moon\\data\\001.0001.npz', 'datasets/moon\\data\\001.0002.npz', ...]
        if data_type == 'train':
            rng.shuffle(
                paths)  # ['datasets/moon\\data\\012.0287.npz', 'datasets/moon\\data\\004.0215.npz', 'datasets/moon\\data\\003.0149.npz', ...]
        if not config.skip_path_filter:
            items = parallel_run(get_frame, paths, desc="filter_by_min_max_frame_batch",
                                 parallel=True)  # [('datasets/moon\\data\\012.0287.npz', 130, 21), ('datasets/moon\\data\\003.0149.npz', 209, 37), ...]

            min_n_frame = hparams['reduction_factor'] * hparams['min_iters']  # 5*30
            max_n_frame = hparams['reduction_factor'] * hparams['max_iters'] - 1  # 5*200 - 5
            # 다음 단계에서 data가 많이 떨어져 나감. 글자수가 짧은 것들이 탈락됨.
            new_items = [(path, n) for path, n, n_tokens in items if
                         min_n_frame <= n <= max_n_frame and n_tokens >= hparams[
                             'min_tokens']]  # [('datasets/moon\\data\\004.0383.npz', 297), ('datasets/moon\\data\\003.0533.npz', 394),...]
            new_paths = [path for path, n in new_items]
            new_n_frames = [n for path, n in new_items]

            hours = frames_to_hours(new_n_frames, hparams)

        else:
            new_paths = paths

        # train용 data와 test용 data로 나눈다.
        if data_type == 'train':
            new_paths = new_paths[:-n_test]  # 끝에 있는 n_test(batch_size)를 제외한 모두
        elif data_type == 'test':
            new_paths = new_paths[-n_test:]  # 끝에 있는 n_test
        else:
            raise Exception(" [!] Unkown data_type: {}".format(data_type))

        path_dict[
            data_dir] = new_paths  # ['datasets/moon\\data\\001.0621.npz', 'datasets/moon\\data\\003.0229.npz', ...]

        log(' [{}] Loaded metadata for {} examples ({:.2f} hours)'.format(data_dir, len(new_n_frames), hours))
        log(' [{}] Max length: {}'.format(data_dir, max(new_n_frames)))
        log(' [{}] Min length: {}'.format(data_dir, min(new_n_frames)))
    return path_dict


class DataFeederTacotron():
    def __init__(self, data_dirs, hparams, config, batches_per_group, data_type, batch_size):
        super(DataFeederTacotron, self).__init__()

        self._hp = hparams
        self._step = 0
        self._offset = defaultdict(lambda: 2)
        self._batches_per_group = batches_per_group

        self.rng = np.random.RandomState(config.random_seed)  # random number generator
        self.data_type = data_type
        self.batch_size = batch_size

        self.min_tokens = hparams['min_tokens']  # 30
        self.min_n_frame = hparams['reduction_factor'] * hparams['min_iters']  # 5*30
        self.max_n_frame = hparams['reduction_factor'] * hparams['max_iters'] - hparams['reduction_factor']  # 5*200 - 5

        # Load metadata:
        self.path_dict = get_path_dict(data_dirs, self._hp, config, self.data_type, n_test=self.batch_size,
                                       rng=self.rng)  # data_dirs: ['datasets/moon\\data']

        self.data_dirs = list(self.path_dict.keys())  # ['datasets/moon\\data']
        self.data_dir_to_id = {data_dir: idx for idx, data_dir in
                               enumerate(self.data_dirs)}  # {'datasets/moon\\data': 0}

        data_weight = {data_dir: 1. for data_dir in self.data_dirs}  # {'datasets/moon\\data': 1.0}

        weight_Z = sum(data_weight.values())

        self.data_ratio = {data_dir: weight / weight_Z for data_dir, weight in data_weight.items()}
        self.is_multi_speaker = len(self.data_dirs) > 1

        log("=" * 40)
        log(pprint.pformat(self.data_ratio, indent=4))
        log("=" * 40)

        if self.data_type == 'test':
            examples = []
            while True:
                for data_dir in self.data_dirs:
                    examples.append(self._get_next_example(data_dir))
                    # print(data_dir, text.sequence_to_text(examples[-1][0], False, True))
                    if len(examples) >= self.batch_size:
                        break
                if len(examples) >= self.batch_size:
                    break

            # test 할 때는 같은 examples로 계속 반복
            self.static_batches = [examples for _ in range(
                self._batches_per_group)]  # [examples, examples,...,examples] <--- 각 example은 2개의 data를 가지고 있다.

        else:
            self.static_batches = None

        # Read a group of examples:
        n = self.batch_size  # 32
        r = self._hp['reduction_factor']  # 4 or 5  min_n_frame,max_n_frame 계산에 사용되었던...
        start = time.time()

        if self.static_batches is not None:  # 'test'에서는 static_batches를 사용한다. static_batches는 init에서 이미 만들어 놓았다.
            batches = self.static_batches
        else:  # 'train'
            examples = []
            for data_dir in self.data_dirs:
                if self._hp['initial_data_greedy']:
                    if self._step < self._hp['initial_phase_step'] and any(
                            "krbook" in data_dir for data_dir in self.data_dirs):
                        data_dir = [data_dir for data_dir in self.data_dirs if "krbook" in data_dir][0]

                if self._step < self._hp['initial_phase_step']:  # 'initial_phase_step': 8000
                    example = [self._get_next_example(data_dir) for _ in range(int(n * self._batches_per_group // len(
                        self.data_dirs)))]  # _batches_per_group 8,또는 32 만큼의 batch data를 만드낟. 각각의 batch size는 2, 또는 32
                else:
                    example = [self._get_next_example(data_dir) for _ in
                               range(int(n * self._batches_per_group * self.data_ratio[data_dir]))]
                examples.extend(example)
            examples.sort(key=lambda x: x[-1])  # 제일 마지막 기준이니까,  len(linear_target) 기준으로 정렬

        self.len = np.shape(examples)[0]
        examples_len = len(examples)
        self.input_data = [examples[i][0] for i in range(examples_len)]
        self.loss_coeff = [examples[i][1] for i in range(examples_len)]
        self.mel_target = [examples[i][2] for i in range(examples_len)]
        self.linear_target = [examples[i][3] for i in range(examples_len)]
        self.stop_token_target = [examples[i][4] for i in range(examples_len)]
        if self.is_multi_speaker:
            self.id = [examples[i][5] for i in range(examples_len)]
            self.linear_target_len = [examples[i][6] for i in range(examples_len)]
        else:
            self.linear_target_len = [examples[i][5] for i in range(examples_len)]
        log('Generated %d batches of size %d in %.03f sec' % (len(examples) // 32, n, time.time() - start))

    def __getitem__(self, item):
        if self.is_multi_speaker:
            return (self.input_data[item], self.loss_coeff[item], self.mel_target[item], self.linear_target[item], \
                   self.stop_token_target[item], self.id[item], self.linear_target_len[item])
        else:
            return (self.input_data[item], self.loss_coeff[item], self.mel_target[item], self.linear_target[item], \
                   self.stop_token_target[item], self.linear_target_len[item])

    def __len__(self):
        return self.len

    def _get_next_example(self, data_dir):
        '''npz 1개를 읽어 처리한다. Loads a single example (input, mel_target, linear_target, cost) from disk'''
        data_paths = self.path_dict[data_dir]

        while True:
            if self._offset[data_dir] >= len(data_paths):
                self._offset[data_dir] = 0

                if self.data_type == 'train':
                    self.rng.shuffle(data_paths)

            data_path = data_paths[self._offset[data_dir]]  # npz파일 1개 선택
            self._offset[data_dir] += 1

            try:
                if os.path.exists(data_path):
                    data = np.load(data_path)  # data속에는 "linear","mel","tokens","loss_coeff"
                else:
                    continue
            except:
                remove_file(data_path)
                continue


            if self.min_n_frame <= data["linear"].shape[0] <= self.max_n_frame and len(
                    data["tokens"]) > self.min_tokens:
                break

        input_data = data['tokens']  # 1-dim
        mel_target = data['mel']

        if 'loss_coeff' in data:
            loss_coeff = data['loss_coeff']
        else:
            loss_coeff = 1
        linear_target = data['linear']
        stop_token_target = np.asarray(
            [0.] * len(mel_target))  # mel_target은 [xx,80]으로 data마다 len이 다르다.  len에 따라 [0,...,0]

        # multi-speaker가 아니면, speaker_id는 넘길 필요 없지만, 현재 구현이 좀 꼬여 있다. 그래서 무조건 넘긴다.
        if self.is_multi_speaker:
            return (input_data, loss_coeff, mel_target, linear_target, stop_token_target, self.data_dir_to_id[data_dir],
                    len(linear_target))
        else:
            return (input_data, loss_coeff, mel_target, linear_target, stop_token_target, len(linear_target))
