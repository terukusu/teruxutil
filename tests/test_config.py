import unittest
from unittest.mock import patch
from teruxutil.config import Config, ConfigStrategy, EnvironmentVariableConfigStrategy, JsonConfigStrategy, SingletonMeta, YamlConfigStrategy


class TestConfig(unittest.TestCase):
    def setUp(self):
        # テストの実行前にシングルトンの状態をリセット
        SingletonMeta.reset_instance(Config)

    def tearDown(self):
        # テストの実行後にもシングルトンの状態をリセット
        SingletonMeta.reset_instance(Config)

    @patch('os.environ', {'TEST_SETTING': '12345', 'TXU_TEST_SETTING': '54321'})
    def test_environment_variable_config_strategy(self):
        """
        環境変数を使用した設定の読み込みをテストする。
        `TEST_SETTING`環境変数が正しく設定されていることを確認する。
        """

        strategy = EnvironmentVariableConfigStrategy()
        config = strategy.load()
        self.assertIsNone(config.get('TEST_SETTING'), 'プレフィックスがTXU_の環境変数は読み込まれないことを確認する。')
        self.assertEqual(config.get('TXU_TEST_SETTING'), '54321', 'プレフィックスがTXU_の環境変数は読み込まれることを確認する。')

    @patch('teruxutil.io.load_json', return_value={'DEBUG': True})
    def test_json_config_strategy(self, mock_load_json):
        """
        JSONファイルを使用した設定の読み込みをテストする。
        JSONファイルから`DEBUG`キーの値がTrueであることを確認する。
        """

        strategy = JsonConfigStrategy('config.json', 'utf-8')
        config = strategy.load()
        mock_load_json.assert_called_with('config.json', 'utf-8')
        self.assertTrue(config['DEBUG'])

    @patch('teruxutil.io.load_yaml', return_value={'config': {'DEBUG': False}})
    def test_yaml_config_strategy(self, mock_load_yaml):
        """
        YAMLファイルを使用した設定の読み込みをテストする。
        YAMLファイルから`DEBUG`キーの値がFalseであることを確認する。
        """

        strategy = YamlConfigStrategy('config.yaml', 'utf-8')
        config = strategy.load()
        mock_load_yaml.assert_called_with('config.yaml', 'utf-8')
        self.assertFalse(config['DEBUG'])

    def test_config_singleton(self):
        """
        Configクラスがシングルトンであることをテストする。
        2つのConfigインスタンスが同一のオブジェクトであることを確認する。
        """

        config1 = Config()
        config2 = Config()
        self.assertIs(config1, config2)

    @patch('teruxutil.config.ConfigStrategy.load', return_value={'DEBUG': 'テスト値です！'})
    def test_config_load(self, mock_strategy):
        """
        ConfigクラスのloadメソッドがConfigのインスタンスを返すことをテストする。
        """

        config = Config(ConfigStrategy()).load()
        self.assertIsInstance(config, Config)

    @patch('teruxutil.config.ConfigStrategy.load', return_value={'DEBUG': 'テスト値です！'})
    def test_config_load_cache(self, mock_strategy):
        """
        Configクラスのloadメソッドがキャッシュを使用することをテストする。
        複数回のload呼び出しで、設定の読み込みが1回だけ行われることを確認する。
        """

        config = Config(ConfigStrategy())
        config.load()
        config.load()

        mock_strategy.assert_called_once()

    @patch('teruxutil.io.load_yaml')
    def test_config_use_like_dict(self, mock_load_yaml):
        """
        Configクラスのloadが一度呼ばれた後は、Config()['key1']のように辞書のように使えることを確認する。
        """
        mock_load_yaml.return_value = {'config': {'key1': 'value1', 'key2': 'value2'}}

        config = Config(YamlConfigStrategy('config.yaml'))
        config.load()

        mock_load_yaml.assert_called_with('config.yaml', None)

        self.assertEqual(Config()['key1'], 'value1', 'Configオブジェクトを辞書的に扱って設定が取得できることを確認する。')

    @patch('teruxutil.io.load_yaml')
    def test_config_use_like_dict_case_no_key(self, mock_load_yaml):
        """
        辞書的に扱ったときに存在しないキーを指定した場合にNoneが返ることをテストする
        """
        mock_load_yaml.return_value = {'config': {'key1': 'value1', 'key2': 'value2'}}

        config = Config(YamlConfigStrategy('config.yaml'))
        config.load()

        mock_load_yaml.assert_called_with('config.yaml', None)

        self.assertEqual(Config()['key3'], None)


if __name__ == '__main__':
    unittest.main()
