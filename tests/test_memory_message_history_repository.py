from collections import defaultdict
from typing import Any

import unittest
from unittest.mock import patch, MagicMock
from teruxutil.chat.message import Message
from teruxutil.chat.memory_message_history_repository import MemoryMessageHistoryRepository


class TestMemoryMessageHistoryRepository(unittest.TestCase):
    """
    MemoryMessageHistoryRepositoryのテストスイート
    """

    @patch('teruxutil.chat.memory_message_history_repository.tiktoken.encoding_for_model')
    @patch('teruxutil.chat.memory_message_history_repository._config', {'chat_history_max_tokens': 2000, 'model': 'gpt-4'})
    def test_save_message(self, mock_encoding_for_model: Any):
        """
        save_message メソッドがメッセージを正しくリストに追加するかテストします。
        このテストでは、保持している履歴のトークン数が上限以下のときにリストに正常に追加されることを検証します。
        """

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [0] * 2000

        mock_encoding_for_model.return_value = mock_tokenizer

        # メッセージを保存し、正しくリストに追加されるかテスト
        message = Message(session_id='session1', role='user', content='Hello, World!')
        repository = MemoryMessageHistoryRepository()
        repository.save_message(message)
        self.assertEqual(len(repository.messages), 1)
        self.assertEqual(repository.messages[0].content, 'Hello, World!')

    @patch('teruxutil.chat.memory_message_history_repository.tiktoken.encoding_for_model')
    @patch('teruxutil.chat.memory_message_history_repository._config', {'chat_history_max_tokens': 2000, 'model': 'gpt-4'})
    def test_save_message_case_over_token(self, mock_encoding_for_model):
        """
        save_message メソッドが、トークン数上限を超えたときに、正しく追加されるかテストします。
        """
        call_count = 0

        def encode_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return [0] * 2000
            elif call_count == 2:
                return [0] * 2001
            return [0] * 2000

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = encode_side_effect

        mock_encoding_for_model.return_value = mock_tokenizer

        repository = MemoryMessageHistoryRepository()

        message = Message(session_id='session1', role='user', content='Hello, World!')
        repository.save_message(message)
        self.assertEqual(len(repository.messages), 1)

        message = Message(session_id='session1', role='user', content='Hello, World!2')
        repository.save_message(message)
        self.assertEqual(len(repository.messages), 1)
        self.assertEqual(repository.messages[0].content, 'Hello, World!2')


    @patch('teruxutil.chat.memory_message_history_repository.tiktoken.encoding_for_model')
    @patch('teruxutil.chat.memory_message_history_repository._config', {'chat_history_max_tokens': 2000, 'model': 'gpt-4'})
    def test_get_all_messages(self, mock_encoding_for_model):
        """
        test_get_all_messages メソッドで全てのメッセージを取得されるかテストします。
        """
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [0] * 1

        mock_encoding_for_model.return_value = mock_tokenizer

        repository = MemoryMessageHistoryRepository()

        repository.messages = [
            Message(session_id='session1', role='user', content='Hello'),
            Message(session_id='session2', role='user', content='World')
        ]
        session_messages = repository.get_all_messages('session1')
        self.assertEqual(len(session_messages), 1)
        self.assertEqual(session_messages[0].content, 'Hello')

    @patch('teruxutil.chat.memory_message_history_repository.tiktoken.encoding_for_model')
    @patch('teruxutil.chat.memory_message_history_repository._config', {'chat_history_max_tokens': 2000, 'model': 'gpt-4'})
    def test_clear_old_messages(self, mock_encoding_for_model):
        """
        トークン数を超えた場合に古いメッセージが削除されるかテスト
        """
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [0] * 1

        mock_encoding_for_model.return_value = mock_tokenizer

        messages = [Message(session_id='session1', role='user', content=f'Message {i}') for i in range(5)]

        repository = MemoryMessageHistoryRepository()

        for msg in messages:
            repository.save_message(msg)

        self.assertLessEqual(len(repository.get_all_messages('session1')), 5)
        self.assertIn('Message 0', [msg.content for msg in repository.get_all_messages('session1')])
        self.assertIn('Message 4', [msg.content for msg in repository.get_all_messages('session1')])
        self.assertNotIn('Message 5', [msg.content for msg in repository.get_all_messages('session1')])


if __name__ == '__main__':
    unittest.main()
