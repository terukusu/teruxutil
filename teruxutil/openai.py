import base64
import json
import uuid

from abc import ABC, abstractmethod
from typing import Callable, Type, Any, List

from openai import AzureOpenAI as OriginalAzureOpenAI, OpenAI as OriginalOpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.audio import Transcription
from pydantic import BaseModel

from teruxutil.config import Config
from .chat import FirestoreMessageHistoryRepository, MessageHistory, message_history

_config = Config()
_DEFAULT_MAX_RETRY_FOR_FORMAT_AI_MESSAGE = 10


class FunctionDefinition(BaseModel):
    """
    FunctionDefinitionクラスは、関数の定義を表すモデルです。
    関数に関連するメタデータ（モデル、説明）および実行ロジックを保持します。

    Attributes:
        model (Type[BaseModel]): 関数の入力値を表すPydanticモデルの型。
        description (str): 関数の説明文。
        function (Callable[[Type[BaseModel]], Any]): `self.model`型のインスタンスを
                                                      引数として受け取り、何らかの処理を行う関数やメソッド。

    Methods:
        to_function_payload(): この関数定義に基づいて関数のペイロードを生成します。
    """

    model: Type[BaseModel]
    description: str
    function: Callable[[Type[BaseModel]], Any]  # 追加されたフィールド

    def to_function_payload(self) -> dict:
        """
        関数のペイロードを生成し、辞書形式で返します。

        Returns:
            dict: 関数のペイロード。このペイロードには、関数の名前('name')、
                  パラメータのスキーマ('parameters')、関数の説明('description')、
                  および実行ロジック('callable')が含まれます。
        """
        return {
            'name': self.function.__name__,
            'parameters': self.model.model_json_schema(),
            'description': self.description,
            # 実行ロジックはペイロードに直接含めることはできないが、
            # ここでその存在を示すか、実行ロジックに関する何らかの情報を含めることができる
        }


class ChatCompletionPayloadBuilder:
    """
    ChatCompletionPayloadBuilderは OpenAIの chat.completion API のためのペイロードを構築するためのクラスです。
    ユーザーメッセージの追加、画像の追加、応答クラスの追加、Function Callingで使用する関数定義の追加、
    そして最終的なペイロードの構築を行うメソッドを提供します。
    """

    def __init__(self, model_name: str, max_tokens: int = None, temperature: float = None):
        """
        ChatCompletionPayloadBuilderクラスのインスタンスを初期化します。

        :param model_name: 使用するモデルの名前。この値は必須です。
        :param max_tokens: リクエストの応答に含める最大トークン数。この値はオプションです。
        :param temperature: 応答の多様性を制御する温度パラメータ。この値はオプションです。
        """

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.payload: dict[str, Any] = {'messages': []}

    def add_user_message(self, prompt):
        """
        ユーザーからのメッセージをペイロードに追加します。

        :param prompt: ユーザーからのメッセージテキスト。
        :return: 自身のインスタンスを返し、メソッドチェーンを可能にします。
        """

        if not prompt:
            return self

        self.payload['messages'].append({
            'role': 'user',
            'content': [{'type': 'text', 'text': prompt}]
        })

        return self

    def add_ai_message(self, prompt:str):
        if not prompt:
            return self

        self.payload['messages'].append({
            'role': 'assistant',
            'content': prompt
        })

        return self

    def add_ai_message_obj(self, message: ChatCompletionMessage):
        if not message:
            return self

        # {'tool_calls': None} を含んでいると非Noneバリデートに引っかかってエラーに成るので除去
        copied_message = {k: v for k, v in message.items() if k != 'tool_calls'}

        self.payload['messages'].append(copied_message)

        return self

    def add_function_response(self, name: str, response: dict[str, Any]):
        self.payload['messages'].append({
            'role': 'function',
            'name': name,
            'content': json.dumps(response, ensure_ascii=False)
        })

    def add_images(self, images):
        """
        画像をペイロードに追加します。

        :param images: (MIMEタイプ, 画像バイナリ)のタプルのリスト。
        :return: 自身のインスタンスを返し、メソッドチェーンを可能にします。
        """

        if not images:
            return self

        for mime_type, image_bin in images:
            image_b64 = base64.b64encode(image_bin).decode('ascii')
            user_message = next((msg for msg in reversed(self.payload['messages']) if msg['role'] == 'user'), None)

            user_message['content'].append({
                'type': 'image_url',
                'image_url': {'url': f'data:{mime_type};base64,{image_b64}'}
            })

        return self

    def add_response_class(self, response_class):
        """
        応答のJSONスキーマ指定するためのクラスをペイロードに追加します。

        :param response_class: 応答のJSONスキーマ指定するためのクラス。
        :return: 自身のインスタンスを返し、メソッドチェーンを可能にします。
        """

        if not response_class:
            return self

        if not self.payload.get('functions'):
            self.payload['functions'] = []

        self.payload['functions'].append({
            'name': 'format_ai_message',
            'description': '最終的なAIの応答をフォーマットするための関数です。最後に必ず実行してください。',
            'parameters': response_class.model_json_schema()
        })

        if not self.payload.get('function_call'):
            self.payload['function_call'] = {'name': 'format_ai_message'}

        return self

    def add_functions(self, function_definitions: List[FunctionDefinition]):
        """
        Function Callingで使用する関数定義をペイロードに追加します。

        :param function_definitions: Function Callingで使用する関数定義のリスト。
        :return: 自身のインスタンスを返し、メソッドチェーンを可能にします。
        """

        if not function_definitions:
            return self

        if not self.payload.get('functions'):
            self.payload['functions'] = []

        self.payload['functions'].extend([func_def.to_function_payload() for func_def in function_definitions])
        self.payload['function_call'] = 'auto'

        return self

    def build(self):
        """
        最終的なペイロードを構築します。

        :return: 構築されたペイロード。
        """

        if self.model_name:
            self.payload['model'] = self.model_name

        if self.max_tokens:
            self.payload['max_tokens'] = self.max_tokens

        if self.temperature is not None:
            self.payload['temperature'] = self.temperature

        return self.payload


class BaseOpenAI(ABC):
    """
    BaseOpenAIは、OpenAI APIを利用した生成AI機能を提供する基底クラスです。
    APIキー、モデル名、最大トークン数、温度、最大リトライ回数を初期設定として提供します。
    """

    def __init__(self, *, api_key: str = None, model_name: str = None,
                 max_tokens: int = None, temperature: float = None, max_retries: int = None,
                 chat_id: str = None, history_enabled: bool = False):
        """
        BaseOpenAIクラスのインスタンスを初期化します。

        :param api_key: OpenAI APIキー。
        :param model_name: 使用するモデルの名前。
        :param max_tokens: 応答に含める最大トークン数。
        :param temperature: 応答の多様性を制御するための温度パラメータ。
        :param max_retries: APIリクエストの最大リトライ回数。
        :param chat_id: チャットID。履歴を有効にする場合は必要。
        :param history_enabled: メッセージ履歴の有効化。
        """

        self.api_key = api_key or _config['api_key']
        self.model_name = model_name or _config['model_name']
        self.max_tokens = max_tokens or _config['max_tokens']
        self.temperature = temperature or _config['temperature']
        self.max_retries = max_retries or _config['max_retries']
        self.history_enabled = history_enabled or _config['history_enabled']
        self.chat_id = chat_id or str(uuid.uuid4())

        if self.history_enabled:
            history_repository = FirestoreMessageHistoryRepository()
            self.history = MessageHistory(session_id=self.chat_id, repository=history_repository)

    def chat_completion(self, prompt: str, *,
                        images: list[(str, bytes)] = None,
                        response_class: Type[BaseModel] = None,
                        functions: List[FunctionDefinition] = None,
                        **kwargs):
        """
        チャット完了機能を利用して、ユーザーからのプロンプトに基づいた応答を生成します。
        画像を指定してた場合には、デフォルトのモデルは設定の「vision_model_name」キーの値になります。

        :param prompt: ユーザーからのプロンプトテキスト。
        :param images: プロンプトに添付する画像のリスト。(MIMEタイプ, 画像バイナリ)のタプル。
        :param response_class: 応答のフォーマットを定義するクラス。
        :param functions: 応答生成に使用される追加の関数定義のリスト。
        :param kwargs: その他のオプションパラメータ。
        :return: OpenAI APIからの応答。
        """

        model_name = kwargs.get('model_name') or self.model_name
        if images:
            model_name = kwargs.get('model_name') or _config['vision_model_name']

        builder = ChatCompletionPayloadBuilder(
            model_name,
            kwargs.get('max_tokens') or self.max_tokens,
            kwargs.get('temperature') or self.temperature
        )

        if self.history_enabled:
            message_list = self.history.get_history()
            for message in message_list:
                if message.role == message_history.ROLE_AI:
                    builder.add_ai_message(message.content)
                elif message.role == message_history.ROLE_USER:
                    builder.add_user_message(message.content)

        builder = (builder.add_user_message(prompt)
                   .add_images(images)
                   .add_response_class(response_class)
                   .add_functions(functions))

        payload = builder.build()

        print(f'payload: {payload}')

        client = self.get_openai_client(kwargs.get('max_retries'))

        max_retry_for_format_ai_message = _DEFAULT_MAX_RETRY_FOR_FORMAT_AI_MESSAGE
        retry_for_format_ai_message = 0

        return_value = None

        while True:
            result = client.chat.completions.create(**payload)

            if not result.choices[0].message.function_call:
                if response_class:
                    if retry_for_format_ai_message < max_retry_for_format_ai_message:
                        print(f"format_ai_message function が実行されていません。リトライします..: retryCount={retry_for_format_ai_message}")
                        retry_for_format_ai_message += 1
                        continue

                    raise Exception("format_ai_message function のリトライ最大回数に到達しましたが、実行されませんでした。")
                else:
                    return_value = result
                    break

            if result.choices[0].message.function_call.name == 'format_ai_message':
                return_value = result
                break

            function_call = result.choices[0].message.function_call
            target_function = next(x.function for x in functions if x.function.__name__ == function_call.name)
            func_args = json.loads(function_call.arguments)

            # function の実行
            func_result = target_function(**func_args)

            builder.add_ai_message_obj(result.choices[0].message.dict())
            builder.add_function_response(target_function.__name__, func_result)

            # payload の更新
            payload = builder.build()

        if self.history_enabled:
            if result.choices[0].message.function_call:
                ai_message = result.choices[0].message.function_call.arguments
            else:
                ai_message = result.choices[0].message.content

            self.history.add_user_message(prompt)
            self.history.add_ai_message(ai_message)

        return return_value

    def audio_transcription(self, input_file_path, *args, **kwargs) -> Transcription:
        client = self.get_openai_client(kwargs.get('max_retries'))
        with open(input_file_path, 'rb') as audio_bytes:
            transcription = client.audio.transcriptions.create(
                model=kwargs.get('model_name') or _config['audio_transcript_model_name'],
                file=audio_bytes
            )

        return transcription

    @abstractmethod
    def get_openai_client(self, max_retries: int = None) -> Transcription:
        """
        OpenAI APIクライアントを取得します。具体的な実装はサブクラスで行います。

        :param max_retries: APIリクエストの最大リトライ回数。
        :return: 設定されたパラメータを持つOpenAI APIクライアント。
        """

        pass


class AzureOpenAI(BaseOpenAI):
    """
    AzureOpenAIは、Azure上でホストされるOpenAI APIを利用するためのクラスです。
    BaseOpenAIクラスを継承し、Azure固有の設定を追加します。
    """

    def __init__(self, *, api_key: str = None, model_name: str = None,
                 max_tokens: int = None, temperature: float = None, max_retries: int = None,
                 api_version: str = None, azure_endpoint: str = None):
        """
        AzureOpenAIクラスのインスタンスを初期化します。

        :param api_key: OpenAI APIキー。
        :param model_name: 使用するモデルの名前。
        :param max_tokens: 応答に含める最大トークン数。
        :param temperature: 応答の多様性を制御するための温度パラメータ。
        :param max_retries: APIリクエストの最大リトライ回数。
        :param api_version: 使用するAPIのバージョン。
        :param azure_endpoint: Azure APIのエンドポイントURL。
        """

        super().__init__(api_key=api_key, model_name=model_name, max_tokens=max_tokens,
                         temperature=temperature, max_retries=max_retries)

        self.api_version = api_version or _config['api_version']
        self.azure_endpoint = azure_endpoint or _config['azure_endpoint']

    def get_openai_client(self, max_retries: int = None):
        """
        Azure上のOpenAI APIクライアントを取得します。

        :param max_retries: APIリクエストの最大リトライ回数。
        :return: Azure API設定を含むOpenAI APIクライアント。
        """

        kwargs = {}
        if max_retries or self.max_retries:
            kwargs['max_retries'] = max_retries or self.max_retries

        client = OriginalAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            **kwargs
        )

        return client


class OpenAI(BaseOpenAI):
    """
    OpenAIは、標準のOpenAI APIを利用するためのクラスです。
    BaseOpenAIクラスを継承します。
    """

    def __init__(self, *, api_key: str = None, model_name: str = None,
                 max_tokens: int = None, temperature: float = None, max_retries: int = None):
        """
        OpenAIクラスのインスタンスを初期化します。

        :param api_key: OpenAI APIキー。
        :param model_name: 使用するモデルの名前。
        :param max_tokens: 応答に含める最大トークン数。
        :param temperature: 応答の多様性を制御するための温度パラメータ。
        :param max_retries: APIリクエストの最大リトライ回数。
        """

        super().__init__(api_key=api_key, model_name=model_name, max_tokens=max_tokens,
                         temperature=temperature, max_retries=max_retries)

    def get_openai_client(self, max_retries: int = None):
        """
        標準のOpenAI APIクライアントを取得します。

        :param max_retries: APIリクエストの最大リトライ回数。
        :return: 標準API設定を含むOpenAI APIクライアント。
        """

        kwargs = {}
        if max_retries or self.max_retries:
            kwargs['max_retries'] = max_retries or self.max_retries

        client = OriginalOpenAI(
            api_key=self.api_key,
            **kwargs
        )

        return client
