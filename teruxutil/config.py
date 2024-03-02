"""
設定管理システム。

このモジュールは、異なるソースから設定情報をロードするための柔軟な方法を提供します。
利用可能なロードストラテジには、環境変数、JSONファイル、YAMLファイルなどがあります。
設定はシングルトンパターンを使用して一度だけロードされ、以降はキャッシュから取得されます。

※ __init__.py にて、環境変数にTXU_CONFIG_FILEが設定されている場合、
　YamlConfigStrategyを使用して設定をロードしシングルトンが初期化されます。

Examples:
    config = Config(JsonConfigStrategy('config.json'))
    settings = config.load()
    print(settings['DEBUG'])

    一度load()した後は、 以下も可能です。
    value1 = Config()['key1']

Available Classes:
- ConfigStrategy: 設定ロードのための基底クラス。
- EnvironmentVariableConfigStrategy: 環境変数から設定をロードするクラス。
- JsonConfigStrategy: JSONファイルから設定をロードするクラス。
- YamlConfigStrategy: YAMLファイルから設定をロードするクラス。
- SingletonMeta: シングルトンパターンのメタクラス。
- Config: 設定情報を保持し、ロードするクラス。

このモジュールは、設定情報を一元管理し、異なるソースからの柔軟な設定ロードを実現することを目的としています。
"""

import os

from typing import Any, Optional

from . import io


class ConfigStrategy:
    """
    設定をロードするためのストラテジのインターフェイス。
    すべての具体的な設定ロードストラテジはこのインターフェイスを実装する必要があります。
    """

    def load(self) -> dict[str, Any]:
        raise NotImplementedError


class EnvironmentVariableConfigStrategy(ConfigStrategy):
    """
    環境変数から設定をロードするストラテジ。
    """

    def load(self):
        """
        環境変数から設定をロードし、それを辞書として返します。

        :return: 環境変数の設定情報を含む辞書。
        """
        config = {key: value for key, value in os.environ.items() if key.startswith('TXU_')}
        return config


class JsonConfigStrategy(ConfigStrategy):
    """
    JSONファイルから設定をロードするストラテジ。
    """

    def __init__(self, file_path: str, encoding: str = None):
        """
        JSONファイルパスを指定してインスタンスを初期化します。

        :param file_path: JSON設定ファイルのパス。
        :param encoding: ファイルのエンコーディング。指定しなければシステムデフォルト。
        """

        self._file_path = file_path
        self._encoding = encoding

    def load(self) -> dict[str, Any]:
        """
        JSONファイルから設定をロードします。

        :return: JSONファイルからロードされた設定情報。
        """

        return io.load_json(self._file_path, self._encoding)


class YamlConfigStrategy(ConfigStrategy):
    """
    YAMLファイルから設定をロードするストラテジ。
    """

    def __init__(self, file_path: str, encoding: str = None):
        """
        YAMLファイルパスを指定してインスタンスを初期化します。

        :param file_path: YAML設定ファイルのパス。
        :param encoding: ファイルのエンコーディング。指定しなければシステムデフォルト。
        """

        self._file_path = file_path
        self._encoding = encoding

    def load(self) -> dict[str, Any]:
        """
        YAMLファイルから設定をロードします。

        :return: YAMLファイルからロードされた設定情報。
        """

        return io.load_yaml(self._file_path, self._encoding)['config']


class SingletonMeta(type):
    """
    シングルトンパターンのためのメタクラス。
    このメタクラスを使用することで、クラスのインスタンスが1つだけ生成されることを保証します。
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def reset_instance(cls, target_class):
        if target_class in cls._instances:
            del cls._instances[target_class]


class Config(metaclass=SingletonMeta):
    """
    設定情報を保持するクラス。
    ストラテジーパターンに基づいた設定のロード方法を実装します。
    一度ロードした設定は内部でキャッシュされます。
    """

    def __init__(self, strategy: Optional[ConfigStrategy] = None):
        self._strategy = strategy
        self._cache = None

    def load(self) -> 'Config':
        """
        ストラテジーを使用して設定をロードします。

        :return: ロードされた設定情報。
        """

        if self._cache is None:
            self._cache = self._strategy.load()

        return self

    def __getitem__(self, key):
        if self._cache is None:
            raise ValueError("Config is not loaded.")
        return self._cache.get(key)
