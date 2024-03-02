import base64

from typing import Type

import unittest

from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from teruxutil.openai import FunctionDefinition, ChatCompletionPayloadBuilder, BaseOpenAI, AzureOpenAI, OpenAI


class TestModel(BaseModel):
    field: str


# テスト用の関数を定義
def test_function(model_instance: Type[BaseModel]) -> str:
    return "processed"


class FunctionDefinitionTests(unittest.TestCase):
    """
    FunctionDefinitionクラスのテストケースを提供します。
    """

    def test_instance_creation(self):
        """
        FunctionDefinitionインスタンスが期待通りに作成されることをテストします。
        """
        func_def = FunctionDefinition(model=TestModel, description="Test function", function=test_function)
        self.assertIsInstance(func_def, FunctionDefinition)
        self.assertEqual(func_def.description, "Test function")
        self.assertIs(func_def.function, test_function)

    def test_to_function_payload(self):
        """
        to_function_payloadメソッドが正しいペイロードを生成することをテストします。
        """
        func_def = FunctionDefinition(model=TestModel, description="Test function", function=test_function)
        payload = func_def.to_function_payload()
        expected_parameters = {
            'properties': {'field': {'title': 'Field', 'type': 'string'}},
            'required': ['field'],
            'title': 'TestModel',
            'type': 'object'
        }
        self.assertEqual(payload['name'], 'test_function')
        self.assertDictEqual(payload['parameters'], expected_parameters)
        self.assertEqual(payload['description'], "Test function")


class ChatCompletionPayloadBuilderTests(unittest.TestCase):
    def test_add_user_message(self):
        """add_user_messageメソッドが正しくユーザーメッセージを追加することを確認する"""
        builder = ChatCompletionPayloadBuilder()
        builder.add_user_message("Hello, world!")
        self.assertEqual(len(builder.payload['messages']), 1)
        self.assertEqual(builder.payload['messages'][0]['content'][0]['text'], "Hello, world!")

    def test_add_images(self):
        """add_imagesメソッドが正しく画像ペイロードを追加することを確認する"""
        builder = ChatCompletionPayloadBuilder()
        builder.add_user_message("Hello, world!")  # ユーザーメッセージを先に追加
        image_data = b"dummy_image_data"
        expected_image_url = f'data:image/png;base64,{base64.b64encode(image_data).decode("ascii")}'

        builder.add_images([("image/png", image_data)])
        self.assertEqual(builder.payload['messages'][0]['content'][1]['type'], 'image_url')
        self.assertEqual(builder.payload['messages'][0]['content'][1]['image_url']['url'], expected_image_url)

    def test_add_response_class(self):
        """add_response_classメソッドが正しくレスポンスクラスを追加することを確認する"""
        builder = ChatCompletionPayloadBuilder()
        response_class = TestModel
        expected_parameters = {'properties': {'field': {'title': 'Field', 'type': 'string'}}, 'required': ['field'], 'title': 'TestModel', 'type': 'object'}

        builder.add_response_class(response_class)
        self.assertEqual(builder.payload['functions'][0]['name'], 'format_ai_response')
        self.assertEqual(builder.payload['functions'][0]['description'], '最終的なAIの応答をフォーマットするための関数です。最後に必ず実行してください。')
        self.assertDictEqual(builder.payload['functions'][0]['parameters'], expected_parameters)
        self.assertDictEqual(builder.payload['function_call'],  {'name': 'format_ai_response'})

    def test_add_response_class_case_after_add_functions(self):
        """add_functionsで関数が事前に追加されていた場合ですも、add_response_classメソッドが正しくレスポンスクラスを追加することを確認する"""
        builder = ChatCompletionPayloadBuilder()

        #  先に関数を追加
        func_def = FunctionDefinition(model=TestModel, description='Test function', function=test_function)
        builder.add_functions([func_def])

        response_class = TestModel
        expected_parameters = {'properties': {'field': {'title': 'Field', 'type': 'string'}}, 'required': ['field'], 'title': 'TestModel', 'type': 'object'}

        builder.add_response_class(response_class)
        self.assertEqual(builder.payload['functions'][1]['name'], 'format_ai_response')
        self.assertEqual(builder.payload['functions'][1]['description'], '最終的なAIの応答をフォーマットするための関数です。最後に必ず実行してください。')
        self.assertDictEqual(builder.payload['functions'][1]['parameters'], expected_parameters)
        self.assertEqual(builder.payload['function_call'],  'auto')

    def test_add_functions(self):
        """add_functionsメソッドが関数定義を正しく追加することを確認する"""
        builder = ChatCompletionPayloadBuilder()
        func_def = FunctionDefinition(model=TestModel, description='Test function', function=test_function)

        expected_parameters = {
            'properties': {'field': {'title': 'Field', 'type': 'string'}},
            'required': ['field'],
            'title': 'TestModel',
            'type': 'object'
        }

        builder.add_functions([func_def])
        self.assertEqual(builder.payload['functions'][0]['name'], 'test_function')
        self.assertDictEqual(builder.payload['functions'][0]['parameters'], expected_parameters)
        self.assertEqual(builder.payload['functions'][0]['description'], "Test function")

    def test_build(self):
        """buildメソッドが正しいペイロードを構築することを確認する"""
        builder = ChatCompletionPayloadBuilder()
        payload = builder.build(model='gpt-4', max_tokens=1000, temperature=0.6)
        self.assertEqual(payload['model'], 'gpt-4')
        self.assertEqual(payload['max_tokens'], 1000)
        self.assertEqual(payload['temperature'], 0.6)

    def test_build_case_zero_temperature(self):
        """buildメソッドが、temperature=0のときに、正しいペイロードを構築することを確認する"""
        builder = ChatCompletionPayloadBuilder()
        payload = builder.build(model='gpt-4', max_tokens=1000, temperature=0)
        self.assertEqual(payload['model'], 'gpt-4')
        self.assertEqual(payload['max_tokens'], 1000)
        self.assertEqual(payload['temperature'], 0)

    def test_build_case_with_only_model(self):
        """buildメソッドが、modelしか指定されていない場合に、正しいペイロードを構築することを確認する"""
        builder = ChatCompletionPayloadBuilder()
        payload = builder.build(model='gpt-4')
        self.assertEqual(payload['model'], 'gpt-4')
        self.assertFalse('max_tokens' in payload)
        self.assertFalse('temperature' in payload)


# BaseOpenAIを継承したテスト用の具象クラスを定義
class ConcreteBaseOpenAI(BaseOpenAI):
    def get_openai_client(self, max_retries: int = None):
        return MagicMock()


class TestBaseOpenAI(unittest.TestCase):
    """
    BaseOpenAIクラスのテストケースです。
    """

    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100, 'temperature': 0.5, 'max_retries': 3})
    def test_init_with_no_param(self):
        """
        BaseOpenAIのコンストラクタが設定値をConfigから正しくロードして初期化するかテストします。
        """
        base_ai = ConcreteBaseOpenAI()
        self.assertEqual(base_ai.api_key, 'test_key')
        self.assertEqual(base_ai.model_name, 'test_model')
        self.assertEqual(base_ai.max_tokens, 100)
        self.assertEqual(base_ai.temperature, 0.5)
        self.assertEqual(base_ai.max_retries, 3)

    @patch('teruxutil.openai._config', {})
    def test_init_case_with_param(self):
        """
        BaseOpenAIのコンストラクタが設定値を引数から正しくロードして初期化するかテストします。
        """
        base_ai = ConcreteBaseOpenAI(api_key='test_key2', model_name='test_model2', max_tokens=1002, temperature=0.52, max_retries=32)
        self.assertEqual(base_ai.api_key, 'test_key2')
        self.assertEqual(base_ai.model_name, 'test_model2')
        self.assertEqual(base_ai.max_tokens, 1002)
        self.assertEqual(base_ai.temperature, 0.52)
        self.assertEqual(base_ai.max_retries, 32)

    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100, 'temperature': 0.5, 'max_retries': 3})
    def test_init_case_with_param_and_config(self):
        """
        BaseOpenAIのコンストラクタで設定値がConfigよりも引数を優先して正しく初期化されるかテストします。
        """
        base_ai = ConcreteBaseOpenAI(api_key='test_key2', model_name='test_model2', max_tokens=1002, temperature=0.52, max_retries=32)
        self.assertEqual(base_ai.api_key, 'test_key2')
        self.assertEqual(base_ai.model_name, 'test_model2')
        self.assertEqual(base_ai.max_tokens, 1002)
        self.assertEqual(base_ai.temperature, 0.52)
        self.assertEqual(base_ai.max_retries, 32)

class TestAzureOpenAI(unittest.TestCase):
    """
    AzureOpenAIクラスのテストケースです。
    """

    @patch('teruxutil.openai.OriginalAzureOpenAI')
    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100,
                                        'temperature': 0.5, 'max_retries': 3, 'api_version': 'test_version',
                                        'azure_endpoint': 'test_endpoint'})
    def test_get_openai_client(self, mock_openai):
        """
        get_openai_clientメソッドが正しくAzureOpenAIクライアントを初期化するかテストします。
        """
        azure_ai = AzureOpenAI()
        client = azure_ai.get_openai_client()
        mock_openai.assert_called_once_with(api_key='test_key', api_version='test_version', azure_endpoint='test_endpoint', max_retries=3)

    @patch('teruxutil.openai.OriginalAzureOpenAI')
    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100,
                                        'temperature': 0.5, 'max_retries': 3, 'api_version': 'test_version',
                                        'azure_endpoint': 'test_endpoint'})
    def test_get_openai_client_case_max_retries_param(self, mock_openai):
        """
        コンストラクタで指定されたmax_retriesの値で、get_openai_clientメソッドが正しくAzureOpenAIクライアントを初期化するかテストします。
        """
        azure_ai = AzureOpenAI(max_retries=5)
        client = azure_ai.get_openai_client()
        mock_openai.assert_called_once_with(api_key='test_key', api_version='test_version', azure_endpoint='test_endpoint', max_retries=5)

    @patch('teruxutil.openai.AzureOpenAI.get_openai_client')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.build')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_functions')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_images')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_response_class')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_user_message')
    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100,
                                        'temperature': 0.5, 'max_retries': 3, 'api_version': 'test_version',
                                        'azure_endpoint': 'test_endpoint'})
    def test_chat_completion(self, mock_add_user_message, mock_add_response_class, mock_add_images, mock_add_functions, mock_build, mock_get_openai_client):
        """
        chat_completionメソッドが期待通りに動作するかテストします。
        """

        setup_mock_chat_completion_payload_builder(
            mock_add_user_message,
            mock_add_response_class,
            mock_add_images,
            mock_add_functions,
            mock_build
        )

        mock_client = MagicMock()
        mock_get_openai_client.return_value = mock_client
        mock_response = {'choices': [{'message': {'content': 'test response'}}]}
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = {
            'messages': [{
                'role': 'user',
                'content': [{'type': 'text', 'text': 'test_prompt'}]
            }],
            'model_name': 'test_model',
            'max_tokens': 100,
            'temperature': 0.5
        }

        # テスト対象メソッドの呼び出し
        azure_ai = AzureOpenAI()
        response = azure_ai.chat_completion('test prompt')

        # モックが期待通りに呼び出されたか検証
        mock_add_user_message.assert_called_once_with('test prompt')
        mock_build.assert_called_once_with('test_model', 100, 0.5)
        mock_get_openai_client.assert_called_once_with(None)
        mock_client.chat.completions.create.assert_called_once_with(**mock_build.return_value)

        # レスポンスが期待通りか検証
        self.assertEqual(response, mock_response)

    @patch('teruxutil.openai.AzureOpenAI.get_openai_client')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.build')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_functions')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_images')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_response_class')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_user_message')
    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100,
                                        'temperature': 0.5, 'max_retries': 3, 'api_version': 'test_version',
                                        'azure_endpoint': 'test_endpoint'})
    def test_chat_completion_with_response_class(self, mock_add_user_message, mock_add_response_class, mock_add_images, mock_add_functions, mock_build, mock_get_openai_client):
        """
        chat_completionメソッドが, response_class を指定して実行されたときにに、期待通りに動作するかテストします。
        """
        # モックの設定
        setup_mock_chat_completion_payload_builder(
            mock_add_user_message,
            mock_add_response_class,
            mock_add_images,
            mock_add_functions,
            mock_build
        )

        mock_client = MagicMock()
        mock_get_openai_client.return_value = mock_client
        mock_response = {'choices': [{'message': {'content': 'test response'}}]}
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = {
            'messages': [{
                'role': 'user',
                'content': [{'type': 'text', 'text': 'test_prompt'}]
            }],
            'functions': [{
                'name': 'format_ai_response',
                'description': '最終的なAIの応答をフォーマットするための関数です。最後に必ず実行してください。',
                'parameters': {
                    'properties': {'field': {'title': 'Field', 'type': 'string'}},
                    'required': ['field'],
                    'title': 'TestModel',
                    'type': 'object'
                }
            }],
            'model_name': 'test_model',
            'max_tokens': 100,
            'temperature': 0.5
        }

        # テスト対象メソッドの呼び出し
        azure_ai = AzureOpenAI()
        response = azure_ai.chat_completion('test prompt', response_class=TestModel)

        # モックが期待通りに呼び出されたか検証
        mock_add_user_message.assert_called_once_with('test prompt')
        mock_add_response_class.assert_called_once_with(TestModel)
        mock_build.assert_called_once_with('test_model', 100, 0.5)
        mock_get_openai_client.assert_called_once_with(None)
        mock_client.chat.completions.create.assert_called_once_with(**mock_build.return_value)

        # レスポンスが期待通りか検証
        self.assertEqual(response, mock_response)

    @patch('teruxutil.openai.AzureOpenAI.get_openai_client')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.build')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_functions')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_images')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_response_class')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_user_message')
    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100,
                                        'temperature': 0.5, 'max_retries': 3, 'api_version': 'test_version',
                                        'azure_endpoint': 'test_endpoint'})
    def test_chat_completion_case_with_images(self, mock_add_user_message, mock_add_response_class, mock_add_images, mock_add_functions, mock_build, mock_get_openai_client):
        """
        chat_completionメソッドが, images を指定して実行されたときにに、期待通りに動作するかテストします。
        """
        # モックの設定
        setup_mock_chat_completion_payload_builder(
            mock_add_user_message,
            mock_add_response_class,
            mock_add_images,
            mock_add_functions,
            mock_build
        )

        mock_client = MagicMock()
        mock_get_openai_client.return_value = mock_client
        mock_response = {'choices': [{'message': {'content': 'test response'}}]}
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = {
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'test_prompt'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64.b64encode(b"test_image_data").decode("ascii")}'
                        }
                    }
                ]
            }],
            'model_name': 'test_model',
            'max_tokens': 100,
            'temperature': 0.5
        }

        # テスト対象メソッドの呼び出し
        azure_ai = AzureOpenAI()
        response = azure_ai.chat_completion('test prompt', images=[('image/jpeg', b'test_image_data')])

        # モックが期待通りに呼び出されたか検証
        mock_add_user_message.assert_called_once_with('test prompt')
        mock_add_images.assert_called_once_with([('image/jpeg', b'test_image_data')])
        mock_build.assert_called_once_with('test_model', 100, 0.5)
        mock_get_openai_client.assert_called_once_with(None)
        mock_client.chat.completions.create.assert_called_once_with(**mock_build.return_value)

        # レスポンスが期待通りか検証
        self.assertEqual(response, mock_response)


    @patch('teruxutil.openai.AzureOpenAI.get_openai_client')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.build')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_functions')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_images')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_response_class')
    @patch('teruxutil.openai.ChatCompletionPayloadBuilder.add_user_message')
    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100,
                                        'temperature': 0.5, 'max_retries': 3, 'api_version': 'test_version',
                                        'azure_endpoint': 'test_endpoint'})
    def test_chat_completion_case_with_functions(self, mock_add_user_message, mock_add_response_class, mock_add_images, mock_add_functions, mock_build, mock_get_openai_client):
        """
        chat_completionメソッドが, functions を指定して実行されたときにに、期待通りに動作するかテストします。
        """
        # モックの設定
        setup_mock_chat_completion_payload_builder(
            mock_add_user_message,
            mock_add_response_class,
            mock_add_images,
            mock_add_functions,
            mock_build
        )

        mock_client = MagicMock()
        mock_get_openai_client.return_value = mock_client
        mock_response = {'choices': [{'message': {'content': 'test response'}}]}
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = {
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'test_prompt'
                    }
                ]
            }],
            'functions': [{
                'name': 'test_function',
                'description': 'Test function',
                'parameters': {
                    'properties': {'field': {'title': 'Field', 'type': 'string'}},
                    'required': ['field'],
                    'title': 'TestModel',
                    'type': 'object'
                }
            }],

            'model_name': 'test_model',
            'max_tokens': 100,
            'temperature': 0.5
        }

        func_def = FunctionDefinition(model=TestModel, description="Test function", function=test_function)
        func_payload = func_def.to_function_payload()

        # テスト対象メソッドの呼び出し
        azure_ai = AzureOpenAI()
        response = azure_ai.chat_completion('test prompt', functions=[func_def])

        # モックが期待通りに呼び出されたか検証
        mock_add_user_message.assert_called_once_with('test prompt')
        mock_add_functions.assert_called_once_with([func_def])
        mock_build.assert_called_once_with('test_model', 100, 0.5)
        mock_get_openai_client.assert_called_once_with(None)
        mock_client.chat.completions.create.assert_called_once_with(**mock_build.return_value)

        # レスポンスが期待通りか検証
        self.assertEqual(response, mock_response)


class TestOpenAI(unittest.TestCase):
    """
    OpenAIクラスのテストケースです。
    """

    @patch('teruxutil.openai.OriginalOpenAI')
    @patch('teruxutil.openai._config', {'api_key': 'test_key', 'model_name': 'test_model', 'max_tokens': 100,
                                        'temperature': 0.5, 'max_retries': 3})
    def test_get_openai_client(self, mock_openai):
        """
        get_openai_clientメソッドが正しくOpenAIクライアントを初期化するかテストします。
        """
        open_ai = OpenAI()
        client = open_ai.get_openai_client()
        mock_openai.assert_called_once_with(api_key='test_key', max_retries=3)


def setup_mock_chat_completion_payload_builder(mock_add_user_message, mock_add_response_class, mock_add_images, mock_add_functions, mock_build):
    mock_instance = MagicMock()

    mock_instance.add_user_message = mock_add_user_message
    mock_instance.add_response_class = mock_add_response_class
    mock_instance.add_images = mock_add_images
    mock_instance.add_functions = mock_add_functions
    mock_instance.build = mock_build

    mock_add_user_message.return_value = mock_instance
    mock_add_response_class.return_value = mock_instance
    mock_add_functions.return_value = mock_instance
    mock_add_images.return_value = mock_instance

    return mock_instance


if __name__ == "__main__":
    unittest.main()
