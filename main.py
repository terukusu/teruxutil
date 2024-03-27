import logging
import os

from pydantic import BaseModel, Field


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/teru/Documents/dev_resources/sa_playground_test-84033.json'
os.environ['TXU_CONFIG_FILE'] = 'config.yaml'

from teruxutil import openai, firestore
from teruxutil.chat import FirestoreMessageHistoryRepository, Message

logging.basicConfig(
    level=logging.DEBUG,
    encoding='utf-8',
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)


class Response(BaseModel):
    text: str = Field(..., description='AIの応答')


# openaiの使い方
def main():

    with open('test_data/sample.png', 'rb') as f:
        image = f.read()

    # 天気取得用関数
    def get_weather(*, location):
        return {
            'location': location,
            'weather': '晴れ'
        }

    # 天気取得用関数の引数を定義するクラス
    class GetWeatherArgument(BaseModel):
        location: str = Field(..., description='場所')

    # AIへ伝える関数情報
    func_def = openai.FunctionDefinition(
        model=GetWeatherArgument,
        description='指定された場所の天気を取得します。',
        function=get_weather
    )

    # client_class = openai.AzureOpenAI
    client_class = openai.OpenAI

    # 一番シンプル
    client = client_class()
    # result = client.chat_completion('こんにちは')

    # カスタムレスポンスクラスを使う
    # client = client_class()
    # result = client.chat_completion('こんにちは', response_class=Response)

    # 画像を使う
    # client = client_class(model_name='gpt-4-vision-preview')
    result = client.chat_completion('この画像を分析して', images=[('image/png', image)])

    # カスタム関数を使う
    # client = client_class()
    # result = client.chat_completion('東京の天気は？そして北海道の天気は？', functions=[func_def], response_class=Response)

    print(result)


#
def main2():

    session_id = 'test_session_id'

    repo = FirestoreMessageHistoryRepository(max_tokens=2000)
    message = Message(
        session_id=session_id,
        role='User',
        content="""
        目的：オブジェクトの「ユーザー向け」の文字列表現を提供します。この表現はより読みやすく、ユーザーが直感的に理解しやすい形式が望ましいです。
デフォルトで呼び出される場面：オブジェクトがprint関数やstr()関数に渡された時に呼び出されます。
        """
    )

    repo.save_message(message)

    all_messages = repo.get_all_messages(session_id)
    all_times = '\n'.join([f'{message.timestamp}' for message in all_messages])
    all_text = '\n'.join([message.content for message in all_messages])

    import tiktoken
    tokenizer = tiktoken.encoding_for_model('gpt-4')
    num_token = len(tokenizer.encode(all_text))

    print(f'messages={len(all_messages)}, tokens={num_token}, all_times={all_times}')


def main3():
    client = openai.OpenAI()
    result = client.chat_completion('こんにちは')

    print(result)


if __name__ == '__main__':
    main()
