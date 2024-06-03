from typing import Any, Optional, Union, Mapping, List

import pandas as pd

from google.cloud import bigquery

from teruxutil.config import Config


_config = Config()


class BigQuery:
    def __init__(self, project_id: str = None, location: str = None):
        self.project_id = project_id or _config['google_cloud_project_id']
        self.location = location or _config['bq_region']
        self.client = bigquery.Client(project=self.project_id, location=self.location)

    def execute_query(
            self,
            bql: str,
            param_types: Optional[List[str]] = None,
            param_values: Optional[List[Any]] = None,
            dtypes: Optional[Mapping[str, Union[str, pd.Series.dtype]]] = None) -> pd.DataFrame:
        """
        BigQuery SQLクエリを実行し、その結果をpandas DataFrameとして返します。

        引数:
            bql (str): 実行するBigQuery SQLクエリ。
            param_types (Optional[List[str]], 任意): クエリパラメータのタイプのリスト。
                各タイプはクエリ内のパラメータに対応します。デフォルトはNoneです。
            param_values (Optional[List[Any]], 任意): クエリパラメータの値のリスト。
                各値は対応するparam_typesに対応します。デフォルトはNoneです。
            dtypes (Optional[Mapping[str, Union[str, pd.Series.dtype]]], 任意): 結果のDataFrameの各列に対するデータ型のマッピング。
                デフォルトはNoneです。

        戻り値:
            pd.DataFrame: クエリ結果を含むpandas DataFrame。
        """

        # TODO 位置パラメータ入りのBQLへ対応する

        query_job = self.client.query(bql)

        kwargs = {}
        if dtypes:
            kwargs['dtypes'] = dtypes

        result = query_job.result().to_dataframe(kwargs)

        return result
