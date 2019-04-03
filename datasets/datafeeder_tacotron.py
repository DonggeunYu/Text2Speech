import threading
import numpy as np
import pprint
import os
import tensorflow as tf
from glob import glob
from collections import defaultdict

from utils.infolog import log
from utils import parallel_run
from utils import parallel_run, remove_file
from utils.audio import frames_to_hours

tf.compat.v1.disable_eager_execution()

def get_frame(path):
    data = np.load(path)
    n_frame = data["linear"].shape[0]
    n_token = len(data["tokens"])
    return (path, n_frame, n_token)

def get_path_dict(data_dirs, hparams, config, data_type, n_test=None, rng=np.random.RandomState(100)):

    path_dict = {}
    for data_dir in data_dirs:
        paths = glob("{}/*.npz".format(data_dir))

        if data_type == 'train':
            rng.shuffle(paths)

        items = parallel_run(get_frame, paths, desc="filter_by_min_max_frame_batch", parallel=True)

        min_n_frame = hparams['reduction_factor'] * hparams['min_iters']  # 5*30
        max_n_frame = hparams['reduction_factor'] * hparams['max_iters'] - hparams['reduction_factor']  # 5*200 - 5

        #글자수를 기준으로 짧거나 길면 짤림
        new_items = [(path, n) for path, n, n_tokens in items if min_n_frame <= n <= max_n_frame and n_tokens >= hparams['min_tokens']] # [('datasets/moon\\data\\004.0383.npz', 297), ('datasets/moon\\data\\003.0533.npz', 394),...]

        new_paths = [path for path, n in new_items]
        new_n_frames = [n for path, n in new_items]

        hours = frames_to_hours(new_n_frames, hparams)

        log(' [{}] Loaded metadata for {} examples ({:.2f} hours)'.format(data_dir, len(new_n_frames), hours))
        log(' [{}] Max length: {}'.format(data_dir, max(new_n_frames)))
        log(' [{}] Min length: {}'.format(data_dir, min(new_n_frames)))

        ''' hccho2 코드인데 왜 이렇게 비효율적으로 만들었는지 모르겠음
            if data_type == 'train':
            new_paths = new_paths[:-n_test]
        elif data_type == 'test':
            new_paths = new_paths[-n_test:]
        '''
        if data_type == 'test':
            new_paths = [new_paths[x] for x in range(n_test)]

        path_dict[data_dir] = new_paths

    return path_dict

class DataFeederTacotron(threading.Thread):
    def __init__(self, coordinator, data_dirs, hparams, config, batches_per_group, data_type, batch_size):
        super(DataFeederTacotron, self).__init__()

        self._coord = coordinator
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

        data_weight = {data_dir: 1. for data_dir in self.data_dirs} # {'datasets/moon\\data': 1.0}

        weight_Z = sum(data_weight.values())

        self.data_ratio = {data_dir: weight / weight_Z for data_dir, weight in data_weight.items()}

        log("="*40)
        log(pprint.pformat(self.data_ratio, indent=4))
        log("="*40)

        self._placeholders = [
            tf.keras.backend.placeholder([None, None], dtype=tf.int32, name='inputs'),
            tf.keras.backend.placeholder([None], dtype=tf.int32, name='input_lengths'),
            tf.keras.backend.placeholder([None], dtype=tf.float32, name='loss_coeff'),
            tf.keras.backend.placeholder([None, None, hparams['num_mels']], dtype=tf.float32, name='mel_targets'),
            tf.keras.backend.placeholder([None, None, hparams['num_freq']], dtype=tf.float32, name='linear_targets'),
        ]

        # Create queue for buffering data:
        dtypes = [tf.int32, tf.int32, tf.float32, tf.float32, tf.float32]

        self.is_multi_speaker = len(self.data_dirs) > 1

        if self.is_multi_speaker:
            self._placeholders.append(tf.keras.backend.placeholder(dtype=tf.int32, shape=[None],
                                                     name='speaker_id'), )  # speaker_id 추가  'inputs'  --> 'speaker_id'로 바꿔야 하지 않나??
            dtypes.append(tf.int32)
        print('a')
        num_workers = os.cpu_count() if self.data_type == 'train' else 1
        queue = tf.queue.FIFOQueue(num_workers, dtypes, name='input_queue')

        self._enqueue_op = queue.enqueue(self._placeholders)

        if self.is_multi_speaker:
            self.inputs, self.input_lengths, self.loss_coeff, self.mel_targets, self.linear_targets, self.speaker_id = queue.dequeue()
        else:
            self.inputs, self.input_lengths, self.loss_coeff, self.mel_targets, self.linear_targets = queue.dequeue()

        self.inputs.set_shape(self._placeholders[0].shape)
        self.input_lengths.set_shape(self._placeholders[1].shape)
        self.loss_coeff.set_shape(self._placeholders[2].shape)
        self.mel_targets.set_shape(self._placeholders[3].shape)
        self.linear_targets.set_shape(self._placeholders[4].shape)

        if self.is_multi_speaker:
            self.speaker_id.set_shape(self._placeholders[5].shape)
        else:
            self.speaker_id = None

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

        return (input_data, loss_coeff, mel_target, linear_target, self.data_dir_to_id[data_dir], len(linear_target))