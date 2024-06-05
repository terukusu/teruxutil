from typing import List

import tiktoken

from .message import Message
from .base_message_history_repository import BaseMessageHistoryRepository

from ..config import Config

_config = Config()


class MemoryMessageHistoryRepository(BaseMessageHistoryRepository):
    """
    メモリ上にメッセージ履歴を保存するリポジトリの実装。

    このリポジトリは、指定されたセッションIDに関連するメッセージをメモリ内に保存し、
    トークン数に基づいて古いメッセージを自動的にクリアします。

    Attributes:
        max_tokens (int): メッセージ履歴内で許可される最大トークン数。デフォルトまたは設定ファイルから読み込まれる。
        encoding_model (str): トークン化に使用されるモデルの名前。デフォルトまたは設定ファイルから読み込まれる。
        messages (List[Message]): 保存されているメッセージのリスト。
    """

    def __init__(self, *, max_tokens: int = None, encoding_model: str = None):
        """
        コンストラクタ。

        Args:
            max_tokens (int, optional): メッセージ履歴内で許可される最大トークン数。
            encoding_model (str, optional): トークン化に使用されるモデルの名前。
        """

        self.max_tokens = max_tokens or _config['chat_history_max_tokens']
        self.encoding_model = encoding_model or _config['openai_model_name']
        self.messages = []

    def save_message(self, message: Message) -> None:
        self.messages.append(message)

    def get_all_messages(self, session_id:str) -> List[Message]:
        return [message for message in self.messages if message.session_id == session_id]

    def clear_old_messages(self, session_id: str) -> None:
        """
        トークン数に基づいて、指定されたセッションIDに関連する古いメッセージをクリアします。

        Args:
            session_id (str): クリアするメッセージのセッションID。
        """

        tokenizer = tiktoken.encoding_for_model(self.encoding_model)
        session_messages = [message for message in self.messages if message.session_id == session_id]

        while True:
            all_message_text = '\n'.join([message.content for message in session_messages])
            if len(tokenizer.encode(all_message_text)) <= self.max_tokens:
                break

            if session_messages:
                oldest_message = session_messages.pop(0)
                self.messages.remove(oldest_message)

    def clear_all_messages(self, session_id: str) -> None:
        """
        指定されたセッションIDに関連するメッセージをクリアします。
        Args:
            session_id (str): クリアするメッセージのセッションID。
        """

        self.messages = [message for message in self.messages if message.session_id != session_id]
