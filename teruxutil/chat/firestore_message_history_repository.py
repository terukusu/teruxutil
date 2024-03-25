import pickle

from typing import Any, List

import tiktoken

from . import Message, BaseMessageHistoryRepository
from ..firestore import Firestore
from ..config import Config

_KEY_SERIALIZED_DATA = 'data'

_config = Config()


class FirestoreMessageHistoryRepository(BaseMessageHistoryRepository):
    def __init__(self, *, collection_name: str = None, max_tokens: int = None, encoding_model: str = None):
        """
        コンストラクタ。

        Args:
            collection_name (str, optional): Cloud Firestore のコレクション名。
            max_tokens (int, optional): メッセージ履歴内で許可される最大トークン数。
            encoding_model (str, optional): トークン化に使用されるモデルの名前。
        """

        self.collection_name = collection_name or _config['chat_history_firestore_collection_name']
        self.max_tokens = max_tokens or _config['chat_history_max_tokens']
        self.encoding_model = encoding_model or _config['model_name']

    def save_message(self, message: Message) -> None:

        def task(doc: dict[str, Any] | None) -> dict[str, Any]:
            messages = []

            if doc:
                data = doc[_KEY_SERIALIZED_DATA]
                messages = pickle.loads(data)

            messages.append(message)

            new_data = pickle.dumps(messages)
            new_doc = {_KEY_SERIALIZED_DATA: new_data}

            return new_doc

        db = self._get_firestore_client()
        db.update_document_in_transaction(message.session_id, task)

    def get_all_messages(self, session_id: str) -> List[Message]:
        db = self._get_firestore_client()
        doc = db.get_document(session_id)
        if doc is None:
            return []

        data = doc[_KEY_SERIALIZED_DATA]
        messages = pickle.loads(data)

        return messages

    def clear_old_messages(self, session_id: str) -> None:
        """
         トークン数に基づいて、指定されたセッションIDに関連する古いメッセージをクリアします。

         Args:
             session_id (str): クリアするメッセージのセッションID。
         """

        tokenizer = tiktoken.encoding_for_model(self.encoding_model)

        def task(doc: dict[str, Any] | None) -> dict[str, Any]:
            session_messages = []

            if doc:
                data = doc[_KEY_SERIALIZED_DATA]
                session_messages = pickle.loads(data)

            while True:
                all_message_text = '\n'.join([message.content for message in session_messages])
                if len(tokenizer.encode(all_message_text)) <= self.max_tokens:
                    break

                if session_messages:
                    session_messages.pop(0)

            new_data = pickle.dumps(session_messages)
            new_doc = {_KEY_SERIALIZED_DATA: new_data}

            return new_doc

        db = self._get_firestore_client()
        db.update_document_in_transaction(session_id, task)

    def clear_all_messages(self, session_id) -> None:
        db = self._get_firestore_client()
        db.delete_document_in_transaction(session_id)

    def _get_firestore_client(self) -> Firestore:
        return Firestore(self.collection_name)
