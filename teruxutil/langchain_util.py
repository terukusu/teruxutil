import logging
import os

from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from .config import Config

_config = Config()


def get_azure_chat(*args, **kwargs):
    default_llm_kwargs = {
        'model_name': _config['openai_model_name'],
        'temperature': _config['openai_temperature'],
        'max_tokens': _config['openai_max_tokens'],
        'openai_api_version': _config['openai_api_version'],
        'openai_api_type': 'azure',
        'azure_endpoint': _config['openai_azure_endpoint'],
        'openai_api_key': _config['openai_api_key'],
    }

    # 引数で指定された値を優先
    new_kwargs = {
        **default_llm_kwargs,
        **kwargs,
    }

    chat = AzureChatOpenAI(**new_kwargs)

    return chat


def get_openai_chat(*args, **kwargs):
    default_llm_kwargs = {
        'model_name': _config['openai_model_name'],
        'temperature': _config['openai_temperature'],
        'max_tokens': _config['openai_max_tokens'],
        'openai_api_key': _config['openai_api_key'],
    }

    # 引数で指定された値を優先
    new_kwargs = {
        **default_llm_kwargs,
        **kwargs,
    }

    chat = ChatOpenAI(**new_kwargs)

    return chat


def get_azure_embeddings(*args, **kwargs):
    embeddings = AzureOpenAIEmbeddings(
        deployment=kwargs.get('openai_embedding_model_name', _config['openai_embedding_model_name']),
        openai_api_version=kwargs.get('openai_api_version', _config['openai_api_version']),
        openai_api_type='azure',
        azure_endpoint=kwargs.get('openai_azure_endpoint', _config['openai_azure_endpoint']),
        openai_api_key=kwargs.get('openai_api_key', _config['openai_api_key'])
    )

    return embeddings


def get_openai_embeddings(*args, **kwargs):
    embeddings = OpenAIEmbeddings(
        deployment=kwargs.get('openai_embedding_model_name',  _config['openai_embedding_model_name']),
        openai_api_key=kwargs.get('openai_api_key',  _config['openai_api_key'])
    )

    return embeddings


def get_vector_store_bigquery(*args, **kwargs):
    from langchain_google_community import BigQueryVectorStore

    embeddings = get_embeddings(kwargs.get('langchain_embedding_type'), **kwargs)

    db = BigQueryVectorStore(
        embedding=embeddings,
        project_id=kwargs.get('google_cloud_project_id', _config['google_cloud_project_id']),
        dataset_name=kwargs.get('langchain_chromadb_folder', _config['langchain_chromadb_folder']),
        table_name=kwargs.get('langchain_vector_store_collection_name', _config['langchain_vector_store_collection_name']),
        location=kwargs.get('bq_region', _config['bq_region'])
    )

    return db


def get_vector_store_chroma(*args, **kwargs):
    import chromadb
    from langchain_community.vectorstores import Chroma

    persist_directory = kwargs.get('langchain_chromadb_folder', _config['langchain_chromadb_folder'])

    client = chromadb.PersistentClient(path=persist_directory)

    embeddings = get_embeddings(kwargs.get('langchain_embedding_type'), **kwargs)

    db = Chroma(
                collection_name=kwargs.get('langchain_vector_store_collection_name', _config['langchain_vector_store_collection_name']),
                embedding_function=embeddings,
                client=client,
                )

    return db


def get_vector_store_pgvector(*args, **kwargs):
    import sqlalchemy
    from langchain_community.vectorstores import PGVector
    from . import cloudsql

    db_config = cloudsql.build_db_config(**kwargs)

    platform = os.getenv('PLATFORM')
    # e.g. 'GCP, GCF, OFFICE'

    base_db_param = {
        'drivername': f'postgresql+{db_config.database_driver}',
        'username': db_config.database_user,
        'password': db_config.database_password,
        'database': db_config.database_name,
    }

    if platform and platform.upper() in ['GCR', 'GCF']:
        # GCRやGCFではUnixドメインソケットを使う
        socket_dir = '/cloudsql/{}:{}:{}'.format(
            db_config.project_id,
            db_config.region,
            db_config.instance_name
        )

        if db_config.database_driver == 'psycopg2':
            query = {'host': f'{socket_dir}'}
        else:
            query = {'unix_sock': f'{socket_dir}/.s.PGSQL.5432'}

        db_param = {
            **base_db_param,
            'query': query,
        }
    else:
        db_param = {
            **base_db_param,
            'host': db_config.database_host,
            'port': db_config.database_port,
        }

    connection_string = sqlalchemy.engine.url.URL.create(**db_param)

    logging.debug(f'connection_string (password is hidden.): {connection_string}')

    embeddings = get_embeddings(kwargs.get('langchain_embedding_type'), **kwargs)

    db = PGVector(
        embedding_function=embeddings,
        collection_name=kwargs.get('langchain_vector_store_collection_name',  _config['langchain_vector_store_collection_name']),
        connection_string=connection_string,
        # TODO: PGVectorのengine_argsをちゃんと設定する
        engine_args={
            'pool_size': 1,
            'max_overflow': 1,
            'pool_timeout': 300,
            'pool_recycle': 1800
        }
    )

    return db


def get_vector_store(vector_store_type: str = None, *args, **kwargs):
    store_type = vector_store_type or _config['langchain_vector_store_type']
    f = _GET_VECTOR_STORE_FUNCTIONS[store_type]

    return f(**kwargs)


def get_embeddings(embedding_type: str = None, *args, **kwargs):
    e_type = embedding_type or _config['langchain_embedding_type']
    f = _GET_EMBEDDING_FUNCTIONS[e_type]

    return f(**kwargs)


def get_chat(chat_type: str = None, *args, **kwargs):
    c_type = chat_type or _config['langchain_chat_type']
    f = _GET_CHAT_FUNCTIONS[c_type]

    return f(**kwargs)


_GET_VECTOR_STORE_FUNCTIONS = {
    'bigquery': get_vector_store_bigquery,
    'pgvector': get_vector_store_pgvector,
    'chroma': get_vector_store_chroma,
}

_GET_EMBEDDING_FUNCTIONS = {
    'azure': get_azure_embeddings,
    'openai': get_openai_embeddings,
}

_GET_CHAT_FUNCTIONS = {
    'azure': get_azure_chat,
    'openai': get_openai_chat,
}
