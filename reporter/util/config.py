import os
from datetime import datetime
from logging import Logger
from pathlib import Path

import toml

from reporter.util.constant import Code


class Span:
    def __init__(self, start: str, end: str):
        self.start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S%z')
        self.end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S%z')


class Config:

    def __init__(self, filename: str):

        self.filename = filename
        config = toml.load(self.filename)

        location = config.get('location', {})
        self.dir_resources = Path(location.get('dir_resources', 'resources'))
        self.dir_logs = Path(location.get('dir_logs', 'logs'))
        self.dir_output = Path(location.get('dir_output', 'output'))

        dataset = config.get('dataset', {})
        self.train_span = Span(*dataset.get('train'))
        self.valid_span = Span(*dataset.get('valid'))
        self.test_span = Span(*dataset.get('test'))
        self.dest_dataset = self.dir_resources / Path(dataset.get('dest_dataset', 'dataset.pkl'))

        train = config.get('train', {})
        self.n_epochs = int(train.get('n_epochs', 10))
        self.batch_size = int(train.get('batch_size', 100))
        self.learning_rate = float(train.get('learning_rate', 1e-4))
        self.token_min_freq = int(train.get('token_min_freq', 1))
        self.rics = sorted(train.get('rics', [Code.N225.value]),
                           key=lambda x: '' if x == Code.N225.value else x)
        self.base_ric = train.get('base_ric', Code.N225.value)
        self.use_standardization = bool(train.get('use_standardization', False))
        self.use_init_token_tag = bool(train.get('use_init_token_tag', True))
        self.patience = int(train.get('patience', 10))

        self.db_uri = config.get('postgres', {}).get('uri')
        self.db_uri_test = config.get('postgres-test', {}).get('uri')

        s3 = config.get('s3', {})
        self.use_aws_env_variables = s3.get('use_aws_env_variables', True)
        self.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.environ.get('AWS_REGION')
        self.aws_profile_name = s3.get('profile_name', 'default')
        self.s3_bucket_name = s3.get('bucket_name')
        self.remote_dir_prices = s3.get('remote_dir_prices')
        self.remote_nikkei_headline_filenames = s3.get('remote_nikkei_headline_filenames', [])

        reuters = config.get('reuters', {})
        self.reuters_username = reuters.get('username', '')
        self.reuters_password = reuters.get('password', '')

        enc = config.get('encoder', {})
        self.enc_hidden_size = int(enc.get('enc_hidden_size', 256))
        self.enc_n_layers = int(enc.get('enc_n_layers', 3))
        self.base_ric_hidden_size = int(enc.get('base_ric_hidden_size', 256))
        self.ric_hidden_size = int(enc.get('ric_hidden_size', 64))
        self.use_dropout = bool(enc.get('use_dropout', True))
        self.word_embed_size = int(enc.get('word_embed_size', 128))
        self.time_embed_size = int(enc.get('time_embed_size', 64))

        attn = config.get('attention', {})
        self.attn_type = attn.get('attn_type', 'scaledot')

        dec = config.get('decoder', {})
        self.dec_hidden_size = int(dec.get('dec_hidden_size', 256))

        self.n_items_per_page = config.get('webapp', {}).get('n_items_per_page', 20)
        self.demo_initial_date = config.get('webapp', {}).get('demo_initial_date', None)

        self.result = dict([(m, Path(p)) for (m, p)
                            in config.get('webapp', {}).get('result', [])])

    def write_log(self, logger: Logger):
        s = '\n'.join(
            [
                f'load configuration from: {self.filename}',
                f'n_epochs: {self.n_epochs}',
                f'batch_size: {self.batch_size}',
                f'learning_rate: {self.learning_rate}',
                f'token_min_freq: {self.token_min_freq}',
                f'rics: {self.rics}',
                f'base_ric: {self.base_ric}',
                f'use_standardization: {self.use_standardization}',
                f'use_init_token_tag: {self.use_init_token_tag}',
                f'patience: {self.patience}',
                f'enc_hidden_size: {self.enc_hidden_size}',
                f'enc_n_layers: {self.enc_n_layers}',
                f'base_ric_hidden_size: {self.base_ric_hidden_size}',
                f'ric_hidden_size: {self.ric_hidden_size}',
                f'use_dropout: {self.use_dropout}',
                f'word_embed_size: {self.word_embed_size}',
                f'time_embed_size: {self.time_embed_size}',
            ]
        )
        logger.info(s)

    @property
    def precise_method_name(self) -> str:
        return ''
