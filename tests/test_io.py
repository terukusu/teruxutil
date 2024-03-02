import unittest
from teruxutil.io import load_text, load_json, load_yaml


class TestIoFunctions(unittest.TestCase):
    def test_load_text(self):
        """テキストファイルを正しく読み込むかテストします。"""
        result = load_text('test_data/test.txt', encoding='utf-8')
        expected = 'これはテストファイルです。'
        self.assertEqual(result, expected)

    def test_load_json(self):
        """JSONファイルを正しく読み込み、Pythonオブジェクトに変換するかテストします。"""
        result = load_json('test_data/test.json', encoding='utf-8')
        expected = {'key': 'テスト値です'}
        self.assertEqual(result, expected)

    def test_load_yaml(self):
        """YAMLファイルを正しく読み込み、Pythonオブジェクトに変換するかテストします。"""
        result = load_yaml('test_data/test.yaml', encoding='utf-8')
        expected = {'config': {'key': 'テスト値です'}}
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
