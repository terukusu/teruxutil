"""
これは実験用のコード。
"""

import logging
import os

from pydantic import BaseModel, Field

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pass to credential file'
os.environ['TXU_CONFIG_FILE'] = 'config.yaml'

from teruxutil import openai, firestore, langchain_util
from teruxutil.chat import FirestoreMessageHistoryRepository, Message
from teruxutil.cloudsql import DatabaseManager
from teruxutil.config import Config

logging.basicConfig(
    level=logging.DEBUG,
    encoding='utf-8',
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)

_config = Config()


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
    result = client.chat_completion('こんにちは')

    # カスタムレスポンスクラスを使う
    # client = client_class()
    # result = client.chat_completion('こんにちは', response_class=Response)

    # 画像を使う
    # client = client_class(model_name='gpt-4-vision-preview')
    # result = client.chat_completion('この画像を分析して', images=[('image/png', image)])

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


def main4():
    client = openai.OpenAI()
    msg = 'いい朝ですね'
    res = client.create_embeddings([msg])
    v = res.data[0].embedding
    print(v)

    with DatabaseManager() as db_manager:
        result = db_manager.execute_query(
            "INSERT INTO embedding (collection_id, content, metadata, vector) VALUES (1, %s, NULL, %s) RETURNING id",
            msg, v)
        print(result)

        return_id = result[0].id

        result = db_manager.execute_query(
            "SELECT id, collection_id, content, created_at, metadata FROM embedding WHERE id = %s", return_id)

        print(result)


def main5():

    vs = langchain_util.get_vector_store(cloud_sql_database_name='test')

    # documents = [
    #     Document(page_content='おはようございます', metadata={'type': 'test', 'page': 1}),
    #     Document(page_content='こんにちは', metadata={'type': 'test', 'page': 2}),
    #     Document(page_content='こんばんは', metadata={'type': 'test', 'page': 3}),
    #     Document(page_content='さようなら', metadata={'type': 'test', 'page': 4}),
    #     Document(page_content='どなたか存じ上げませんが', metadata={'type': 'test', 'page': 5}),
    # ]
    # vs.add_documents(documents)

    result = vs.similarity_search('いい朝ですね。', k=3)

    print(result)


def main6():
    db = langchain_util.get_vector_store()

    db.add_texts([
        'おはようございます',
        'こんにちは',
        'こんばんは',
        'さようなら',
        'なんだなんだ君は'])

    result = db.similarity_search('いい朝ですね')

    print(result)


if __name__ == '__main__':
    main6()
