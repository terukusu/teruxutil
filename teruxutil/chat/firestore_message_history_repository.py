from typing import List

from .message import Message
from .base_message_history_repository import BaseMessageHistoryRepository


class FirestoreMessageHistoryRepository(BaseMessageHistoryRepository):
    def save_message(self, message: Message) -> None:
        # Firestoreへの保存処理を実装
        pass

    def get_all_messages(self, session_id: str) -> List[Message]:
        # Firestoreからの全メッセージ取得処理を実装
        pass

    def clear_old_messages(self, session_id: str) -> None:
        pass
