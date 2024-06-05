import logging

from typing import Any, Optional, Union, Mapping, List

import pandas as pd

from google.cloud import bigquery
from google.cloud.bigquery import ScalarQueryParameterType, StructQueryParameterType

from teruxutil.config import Config


_config = Config()


class ScalarQueryParameter:
    def __init__(self, name: Optional[str], value_type: str, value: Any):
        """
        ScalarQueryParameterのコンストラクタ。

        Args:
            name (Optional[str]): パラメータの名前。省略可能。省略した場合はクエリ中の「？」に順番に対応します。
            value_type (str): パラメータの型。
            value (Any): パラメータの値。
        """

        self.name = name
        self.value_type = value_type
        self.value = value

    def to_bigquery_param(self):
        return bigquery.ScalarQueryParameter(self.name, self.value_type, self.value)

    def __str__(self):
        return f"ScalarQueryParameter(name={self.name}, value_type={self.value_type}, value={self.value})"

    def __repr__(self):
        return f"ScalarQueryParameter(name={self.name}, value_type={self.value_type}, value={self.value})"


class ArrayQueryParameter:
    def __init__(self, name: Optional[str], value_type: Union[str, ScalarQueryParameterType, StructQueryParameterType], value: List[Any]):
        """
        ArrayQueryParameterのコンストラクタ。

        Args:
            name (Optional[str]): パラメータの名前。省略可能。省略した場合はクエリ中の「？」に順番に対応します。
            value_type (str): パラメータの型。
            value (List[Any]): パラメータの値のリスト。
        """

        self.name = name
        self.value_type = value_type
        self.value = value

    def to_bigquery_param(self):
        return bigquery.ArrayQueryParameter(self.name, self.value_type, self.value)

    def __str__(self):
        return f"ArrayQueryParameter(name={self.name}, value_type={self.value_type}, value={self.value})"

    def __repr__(self):
        return f"ArrayQueryParameter(name={self.name}, value_type={self.value_type}, value={self.value})"


class BigQuery:
    def __init__(self, project_id: str = None, location: str = None):
        self.project_id = project_id or _config['google_cloud_project_id']
        self.location = location or _config['bq_region']
        self.client = bigquery.Client(project=self.project_id, location=self.location)

    def execute_query(
            self,
            query: str,
            query_params: Optional[List[Union[ScalarQueryParameter, ArrayQueryParameter]]] = None,
            dtypes: Optional[Mapping[str, Union[str, pd.Series.dtype]]] = None) -> pd.DataFrame:
        """
        BigQuery SQLクエリを実行し、その結果をpandas DataFrameとして返します。

        引数:
            query (str): 実行するBigQuery SQLクエリ。
            query_params (Optional[List[Union[ScalarQueryParameter, ArrayQueryParameter]]], 任意): クエリパラメータのリスト。
            dtypes (Optional[Mapping[str, Union[str, pd.Series.dtype]]], 任意): 結果のDataFrameの各列に対するデータ型のマッピング。
                デフォルトはNoneです。

        戻り値:
            pd.DataFrame: クエリ結果を含むpandas DataFrame。
        """

        logging.info(f'Query: {query}, query_params: {query_params}')

        query_params = [param.to_bigquery_param() for param in query_params] if query_params else []

        job_config = bigquery.QueryJobConfig(
            query_parameters=query_params
        )

        query_job = self.client.query(query, job_config=job_config)

        kwargs = {}
        if dtypes:
            kwargs['dtypes'] = dtypes

        result = query_job.result().to_dataframe(kwargs)

        return result
