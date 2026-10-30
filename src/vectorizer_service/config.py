from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding model settings
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("EMBEDDING_MODEL_NAME"),
    )
    vector_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("VECTOR_SIZE"),
    )

    # gRPC server settings
    grpc_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("GRPC_HOST"),
    )
    grpc_port: int = Field(
        default=50051,
        validation_alias=AliasChoices("GRPC_PORT"),
    )
    grpc_max_workers: int = Field(
        default=4,
        validation_alias=AliasChoices("GRPC_MAX_WORKERS"),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
