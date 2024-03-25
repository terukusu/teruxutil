import unittest
from unittest.mock import patch, MagicMock
from teruxutil.firestore import Firestore
from teruxutil import firestore


class TestFirestore(unittest.TestCase):
    def setUp(self):
        # _config をモック化
        self.mock_config_patch = patch('teruxutil.firestore._config', {'cloud_firestore_database_name': '(default)',
                                                                       'cloud_firestore_collection_name': 'my_collection'})
        self.mock_config = self.mock_config_patch.start()
        self.addCleanup(self.mock_config_patch.stop)

        # Firestore クライアントをモック化
        self.mock_firestore_client_patch = patch('teruxutil.firestore.firestore.Client')
        self.mock_firestore_client = self.mock_firestore_client_patch.start()
        self.addCleanup(self.mock_firestore_client_patch.stop)

        self.firestore = Firestore()

    def test_get_firestore_client(self):
        self.firestore.get_firestore_client()
        self.mock_firestore_client.assert_called_once_with(database='(default)')

    def test_get_collection(self):
        self.mock_firestore_client.return_value.collection.return_value = MagicMock()
        collection = self.firestore.get_collection()
        self.mock_firestore_client.return_value.collection.assert_called_once_with('my_collection')
        self.assertTrue(collection)

    @patch('teruxutil.firestore.Firestore.get_document_ref')
    def test_get_document(self, mock_get_document_ref):
        doc_snapshot = MagicMock()
        doc_snapshot.exists = True
        doc_snapshot.to_dict.return_value = {'key': 'value'}
        mock_get_document_ref.return_value.get.return_value = doc_snapshot

        result = self.firestore.get_document('doc_id')
        self.assertEqual(result, {'key': 'value'})

    @patch('teruxutil.firestore.Firestore.get_document_ref')
    def test_set_document(self, mock_get_document_ref):
        doc_ref = MagicMock()
        col_ref = MagicMock()
        col_ref.document.return_value = doc_ref

        mock_get_document_ref.return_value = col_ref

        self.firestore.set_document('doc_id', {'key': 'value'})
        doc_ref.set.assert_called_once_with({'key': 'value'})


if __name__ == '__main__':
    unittest.main()
