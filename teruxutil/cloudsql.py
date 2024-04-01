import logging
import math
import os

from collections import namedtuple

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel

from .config import Config

_config = Config()

# バルクインサート時のプレースホルダの最大個数
_MAX_QUERY_ARGS = 30000


class DatabaseConfig(BaseModel):
    """
    データベース設定を表すPydanticモデル。

    Attributes:
        project_id (str): Google CloudプロジェクトID。
        region (str): Cloud SQLのリージョン。
        instance_name (str): Cloud SQLインスタンス名。
        database_user (str): データベースユーザー名。
        database_password (str): データベースパスワード。
        database_name (str): データベース名。
        database_host (str): データベースホスト。
        database_port (int): データベースポート。
        database_driver (str): データベースドライバー。
    """

    project_id: str
    region: str
    instance_name: str
    database_user: str
    database_password: str
    database_name: str
    database_host: str
    database_port: int
    database_driver: str


def build_db_config(**kwargs) -> DatabaseConfig:
    """
    Configオブジェクトからデータベース設定を構築する。
    キーワード引数を指定することでデフォルト設定をオーバーライドできる。

    Args:
        google_cloud_project_id (str): Google CloudのプロジェクトID。指定されていない場合、_configから読み込む。
        cloud_sql_region (str): Cloud SQLのリージョン。指定されていない場合、_configから読み込む。
        cloud_sql_instance_name (str): Cloud SQLインスタンスの名前。指定されていない場合、_configから読み込む。
        cloud_sql_database_user (str): データベースのユーザー名。指定されていない場合、_configから読み込む。
        cloud_sql_database_password (str): データベースのパスワード。指定されていない場合、_configから読み込む。
        cloud_sql_database_name (str): データベースの名前。指定されていない場合、_configから読み込む。
        cloud_sql_host (str): データベースのホスト。指定されていない場合、_configから読み込む。
        cloud_sql_port (int): データベースのポート番号。指定されていない場合、_configから読み込む。
        cloud_sql_driver (str): データベースのドライバー。指定されていない場合、_configから読み込む。

    Returns:
        DatabaseConfig: データベース設定のオブジェクト。
    """

    db_config = DatabaseConfig(
        project_id=kwargs.get('google_cloud_project_id', _config['google_cloud_project_id']),
        region=kwargs.get('cloud_sql_region', _config['cloud_sql_region']),
        instance_name=kwargs.get('cloud_sql_instance_name', _config['cloud_sql_instance_name']),
        database_user=kwargs.get('cloud_sql_database_user', _config['cloud_sql_database_user']),
        database_password=kwargs.get('cloud_sql_database_password', _config['cloud_sql_database_password']),
        database_name=kwargs.get('cloud_sql_database_name', _config['cloud_sql_database_name']),
        database_host=kwargs.get('cloud_sql_host', _config['cloud_sql_host']),
        database_port=kwargs.get('cloud_sql_port', _config['cloud_sql_port']),
        database_driver=kwargs.get('cloud_sql_driver', _config['cloud_sql_driver']),
    )

    return db_config


class DatabaseManager:
    """
    データベース操作を管理するクラス。

    環境変数を使用してCloud SQLへの接続情報を設定する。GCF(Google Cloud Functions)やGCR(Google Cloud Run)環境では
    Unixドメインソケットを通じて接続する。

    Attributes:
        dbconfig (DatabaseConfig): データベース設定。
    """

    def __init__(self, *, dbconfig: DatabaseConfig = None):
        """
        DatabaseManagerインスタンスを初期化する。

        Args:
            dbconfig (DatabaseConfig, optional): データベース設定。指定されない場合はbuild_db_configにより生成される。
        """
        self.dbconfig = dbconfig or build_db_config()
        self._conn = None

    def transaction(self, *, isolation=None, readonly=False, deferrable=False):
        return self.conn.transaction(isolation=isolation, readonly=readonly, deferrable=deferrable)

    def _get_connection(self):
        """
        データベースへの接続を確立する内部メソッド。

        環境変数 'PLATFORM' の値に応じて、接続方法を選択します。
        Google Cloud Functions (GCF) や Google Cloud Run (GCR) で実行される場合、Unix ドメインソケットを使用して Cloud SQL インスタンスに接続します。
        それ以外の環境では、標準のTCP接続を使用します。
        """

        if self._conn and not self._conn.is_closed():
            return self._conn

        db_config = build_db_config()

        pf = os.getenv('PLATFORM')

        # GCFやGCRではUnixドメインソケットを使う
        if pf in ['GCF', 'GCR']:
            socket_dir = '/cloudsql/{}:{}:{}'.format(
                db_config.project_id,
                db_config.region,
                db_config.instance_name
            )

            host = socket_dir
            # ドライバ(psycopg2)が↓この値を使ってこのUnixドメインソケットパスを生成する → /cloudsql/INSTANCE_CONNECTION_NAME/.s.PGSQL.5432
            port = "5432"
        else:
            host = db_config.database_host
            port = db_config.database_port

        connection_string = f"host={host} port={port} dbname={db_config.database_name} " \
                            f"user={db_config.database_user} password={db_config.database_password}"

        self._conn = psycopg2.connect(connection_string)
        register_vector(self._conn)

    def execute_query(self, query: str, *args, record_class=None) -> list:
        """
        SQLクエリを実行し、結果を返す。

        Args:
            query (str): SQLクエリ文字列。
            *args: クエリに渡すパラメータ。
            record_class: 結果をマッピングするクラス。指定しない場合はnamedtupleが使用される。

        Returns:
            list: クエリ結果のリスト。
        """

        with self._conn.cursor() as cur:
            cur.execute(query, args)
            try:
                result = cur.fetchall()
                columns = [col.name for col in cur.description]
            except psycopg2.ProgrammingError as e:
                logging.debug(f'An expected error occurred, but the process will continue as planned: error={e}')
                result = []

        if not result:
            return result

        if record_class:
            result_list = [record_class(**dict(zip(columns, row))) for row in result]
        else:
            Record = namedtuple('Record', columns)
            result_list = [Record(*row) for row in result]

        return result_list

    # TODO 複数行挿入を最高速のCOPYコマンドで実装する. insert_raws メソッドとか

    def insert_rows_with_returning(self, table: str, *args, records: list[list], columns: list[str], return_columns: list[str] = None, timeout=None) -> list:
        if not records:
            # レコードが空のときは何もしない
            return []

        # SQLに含められる限界数を超えないような record 数を計算。そのrecord数ずつに分割
        chunk_size = math.floor(_MAX_QUERY_ARGS/len(columns))
        chunked_records = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

        column_list_sql = ','.join([f'"{col}"' for col in columns])

        if return_columns:
            return_column_list_sql = 'RETURNING {}'.format(','.join([f'"{col}"' for col in return_columns]))
        else:
            return_column_list_sql = ''

        returnings = []

        with self._conn.cursor() as cur:
            for chunk in chunked_records:
                query = f'INSERT INTO "{table}" ({column_list_sql}) VALUES %s {return_column_list_sql}'

                execute_values(cur, query, chunk)

                if return_columns:
                    result = cur.fetchall()
                    columns = [col.name for col in cur.description]

                    Record = namedtuple('Record', columns)
                    returnings.extend([Record(*row) for row in result])

        return returnings

    def close(self, exc_type, exc, tb):
        if self._conn:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()

            self._conn.close()

    def is_closed(self):
        return not self._conn or self._conn.is_closed()

    def __enter__(self):
        self.conn = self._get_connection()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type, exc, tb)
