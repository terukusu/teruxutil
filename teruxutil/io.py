"""
ファイル読み込みユーティリティモジュール。

このモジュールは、異なる形式のファイルからデータを読み込み、Pythonのデータ構造に変換するための関数を提供します。
テキストファイル、JSONファイル、およびYAMLファイルをサポートしています。各関数は、指定されたファイルパスからデータを読み込み、
それを適切なPythonのデータ型に変換して返します。ファイルのエンコーディングは指定可能であり、指定がない場合はシステムのデフォルトエンコーディングを使用します。

主な関数:
- load_text: テキストファイルを読み込み、その内容を文字列として返します。
- load_json: JSONファイルを読み込み、その内容をPythonの辞書またはリストとして返します。
- load_yaml: YAMLファイルを読み込み、その内容をPythonの辞書またはリストとして返します。

使用例:
    text_data = load_text('example.txt')
    json_data = load_json('example.json')
    yaml_data = load_yaml('example.yaml')

このモジュールは、設定ファイルやデータファイルの読み込みに便利です。
"""

import json

import yaml

from typing import Any


def load_text(file_path: str, encoding=None) -> str:
    """
    指定されたファイルパスからテキストを読み込む。
    :param file_path: テキストファイルのパス。
    :param encoding: ファイルのエンコーディング。指定がない場合はシステムデフォルトを使用。
    :return: ファイルの内容。
    """

    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()


def load_json(file_path: str, encoding=None) -> Any:
    """
    指定されたファイルパスからJSONを読み込み、Pythonのデータ構造に変換する。
    内部で `load_text` を使用してテキストを読み込む。
    :param file_path: JSONファイルのパス。
    :param encoding: ファイルのエンコーディング。指定がない場合はシステムデフォルトを使用。
    :return: JSONデータをパースしたPythonのデータ構造。
    """

    text = load_text(file_path, encoding=encoding)
    return json.loads(text)


def load_yaml(file_path: str, encoding=None) -> Any:
    """
    指定されたファイルパスからYAMLを読み込み、Pythonのデータ構造に変換する。
    内部で `load_text` を使用してテキストを読み込む。
    :param file_path: YAMLファイルのパス。
    :param encoding: ファイルのエンコーディング。指定がない場合はシステムデフォルトを使用。
    :return: YAMLデータをパースしたPythonのデータ構造。
    """

    text = load_text(file_path, encoding=encoding)
    return yaml.safe_load(text)
