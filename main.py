import logging

from pydantic import BaseModel, Field

from teruxutil import openai

logging.basicConfig(
    level=logging.DEBUG,
    encoding='utf-8',
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)


class Response(BaseModel):
    text: str = Field(..., description='AIの応答')


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
    # client = client_class()
    # result = client.chat_completion('こんにちは')

    # カスタムレスポンスクラスを使う
    # client = client_class()
    # result = client.chat_completion('こんにちは', response_class=Response)

    # 画像を使う
    # client = client_class(model_name='gpt-4-vision-preview')
    # result = client.chat_completion('この画像を分析して', images=[('image/png', image)])

    # カスタム関数を使う
    client = client_class()
    result = client.chat_completion('東京の天気は？そして北海道の天気は？', functions=[func_def], response_class=Response)

    print(result)


if __name__ == '__main__':
    main()
