import uuid
from typing import Optional

from .. import util


class Message:
    def __init__(self, *, id_: str = None, session_id: str, role: str, content: str, timestamp: Optional[str] = None):
        self.id = id_ or str(uuid.uuid4())
        self.session_id = session_id
        self.role = role
        self.content = content
        self.timestamp = timestamp or util.get_now_jst()

    def dict(self):
        return {'id': self.id, 'session_id': self.session_id, 'role': self.role, 'content': self.content}

    def __repr__(self):
        # クラス名とプロパティ名=値のペアを文字列化
        properties_str = ', '.join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({properties_str})"
