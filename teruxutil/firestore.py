"""
Cloud Firestore を簡単に使えるようにしたユーティリティクラス。
設定値の読み書き、ドキュメントの取得と更新など、基本的な操作を手軽に行えます。
"""

from typing import Any, Callable, Dict

from google.cloud import firestore_v1 as firestore

from .config import Config
from .util import get_now_jst

_config = Config()


class Firestore:
    collection_name: str
    _firestore_client: firestore.Client | None
    _listeners: Dict[str, list[firestore.watch.Watch]] = {}

    def __init__(self, collection_name: str = None):
        """
        Firestore クラスのコンストラクタ。
        コレクション名が指定されていない場合は、設定からデフォルトのコレクション名を使用します。

        Args:
            collection_name (str): Cloud Firestore のコレクション名。
        """

        self.collection_name = collection_name or _config['cloud_firestore_collection_name']
        self._firestore_client = None

    def get_firestore_client(self) -> firestore.Client:
        if not self._firestore_client:
            self._firestore_client = firestore.Client(
                database=_config['cloud_firestore_database_name']
            )

        return self._firestore_client

    def get_collection(self) -> firestore.CollectionReference:
        """
        指定されたコレクションへの参照を取得します。

        Returns:
            firestore.CollectionReference: Firestore コレクションへの参照。
        """

        client = self.get_firestore_client()
        return client.collection(self.collection_name)

    def get_document(self, key: str) -> dict[str, Any] | None:
        """
        指定されたキーのドキュメントを取得します。

        Args:
            key (str): ドキュメントを取得するキー。

        Returns:
            dict[str, Any] | None: ドキュメントのデータ、もしくはキーが存在しない場合は None。
        """

        doc_ref = self.get_document_ref(key)
        doc = doc_ref.get()

        if not doc.exists:
            return None

        return doc.to_dict()

    def get_document_ref(self, key: str) -> firestore.DocumentReference:
        """
        指定されたキーのドキュメントへの参照を取得します。

        Args:
            key (str): ドキュメント参照を取得するキー。

        Returns:
            firestore.DocumentReference: 指定されたキーのドキュメントへの参照。
        """

        doc_ref = self.get_collection().document(key)
        return doc_ref

    def set_document(self, key: str, document: dict[str, Any]) -> None:
        """
        指定されたキーにドキュメントをセットします。

        Args:
            key (str): ドキュメントをセットするキー。
            document (dict[str, Any]): セットするドキュメントのデータ。
        """

        doc_ref = self.get_collection().document(key)
        doc_ref.set(document)

        return

    def delete_document(self, key: str) -> None:
        """
        指定されたキーのドキュメントを削除します。

        Args:
            key (str): 削除するドキュメントのキー。
        """

        doc_ref = self.get_collection().document(key)
        doc_ref.delete()

        return

    def delete_document_in_transaction(self, key: str) -> None:
        """
        トランザクション内で指定されたドキュメントを削除します。

        Args:
            key (str): 削除するドキュメントのキー。
        """

        transaction = self.get_firestore_client().transaction()
        doc_ref = self.get_collection().document(key)
        snapshot = doc_ref.get(transaction=transaction)

        if snapshot.exists:
            transaction.delete(doc_ref)

    def update_field_in_transaction(self, key: str, field: str, value: Any) -> None:
        """
        トランザクション内で指定されたフィールドを更新します。

        Args:
            key (str): 更新するドキュメントのキー。
            field (str): 更新するフィールド名。
            value (Any): セットする値。
        """

        transaction = self.get_firestore_client().transaction()
        return Firestore.update_field_in_transaction_internal(transaction, self, key, field, value)

    @staticmethod
    @firestore.transactional
    def update_field_in_transaction_internal(transaction: firestore.Transaction, instance: 'Firestore', key: str, field: str, value: Any) -> None:
        """
        トランザクション内で指定されたフィールドを更新する内部メソッド。

        Args:
            transaction (firestore.Transaction): Firestore トランザクション。
            instance (Firestore): Firestore インスタンス。
            key (str): 更新するドキュメントのキー。
            field (str): 更新するフィールド名。
            value (Any): セットする値。
        """

        doc_ref = instance.get_collection().document(key)

        transaction.update(doc_ref, {field: value})

    def update_document_in_transaction(self, key: str, update_function: Callable[[dict[str: Any]], dict[str: Any]]) -> None:
        """
        トランザクション内でドキュメントを更新する関数を使用してドキュメントを更新します。

        Args:
            key (str): 更新するドキュメントのキー。
            update_function (Callable[[dict[str: Any]], dict[str: Any]]): 現在のドキュメントデータを引数にとり、更新されたデータを返す関数。
        """
        transaction = self.get_firestore_client().transaction()
        return Firestore.update_document_in_transaction_internal(transaction, self, key, update_function)

    @staticmethod
    @firestore.transactional
    def update_document_in_transaction_internal(transaction: firestore.Transaction, instance: 'Firestore', key: str, update_function: Callable[[dict[str: Any]], dict[str: Any]]) -> None:
        """
        トランザクション内でドキュメントを更新する内部メソッド。

        Args:
            transaction (firestore.Transaction): Firestore トランザクション。
            instance (Firestore): Firestore インスタンス。
            key (str): 更新するドキュメントのキー。
            update_function (Callable[[dict[str: Any]], dict[str: Any]]): 現在のドキュメントデータを引数にとり、更新されたデータを返す関数。
        """

        doc_ref = instance.get_collection().document(key)
        snapshot = doc_ref.get(transaction=transaction)

        doc = None
        if snapshot.exists:
            doc = snapshot.to_dict()

        updated_data = update_function(doc)
        transaction.set(doc_ref, updated_data)

    def set_update_listener(self, key: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """
        指定されたキーのドキュメントにアップデートリスナーをセットします。ドキュメントが更新されるたびに、
        指定されたコールバック関数が呼び出されます。

        Args:
            key (str): アップデートリスナーをセットするドキュメントのキー。
            callback (Callable[[dict[str, Any]], None]): ドキュメントが更新された際に呼び出されるコールバック関数。
        """
        def on_snapshot(doc_snapshot, changes, read_time):
            # TODO doc_snapshot と changesの違いを確認して処理を適切にする
            for change in changes:
                doc = change.document.to_dict()
                callback(doc)

        doc_ref = self.get_collection().document(key)
        watch = doc_ref.on_snapshot(on_snapshot)

        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(watch)

    def remove_update_listeners(self, key: str) -> None:
        """
        指定されたキーのドキュメントに設定されている全てのアップデートリスナーを破棄します。

        Args:
            key (str): アップデートリスナーを破棄するドキュメントのキー。
        """
        if key in self._listeners:
            for watch in self._listeners[key]:
                watch.unsubscribe()
            del self._listeners[key]
