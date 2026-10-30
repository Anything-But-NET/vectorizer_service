import logging
import signal
from concurrent import futures

import grpc

from vectorizer_service.config import get_settings
from vectorizer_service.grpc_gen import vectorizer_pb2_grpc
from vectorizer_service.servicer import MarkdownVectorizerServicer
from vectorizer_service.services.transform_service import VectorizationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


MAX_MESSAGE_LENGTH = 50 * 1024 * 1024  # 50 MB
GRPC_SERVER_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
]


def create_server(
    vectorizer: VectorizationService | None = None,
    max_workers: int | None = None,
    options: list[tuple[str, int]] | None = None,
) -> grpc.Server:
    settings = get_settings()
    workers = max_workers or settings.grpc_max_workers
    server_options = options if options is not None else GRPC_SERVER_OPTIONS
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        options=server_options,
    )
    servicer = MarkdownVectorizerServicer(vectorizer=vectorizer)
    vectorizer_pb2_grpc.add_MarkdownVectorizerServicer_to_server(servicer, server)
    return server


def serve() -> None:
    settings = get_settings()
    logger.info(
        "Initializing VectorizationService with model '%s'...",
        settings.embedding_model_name,
    )
    vectorizer = VectorizationService(model_name=settings.embedding_model_name)

    server = create_server(vectorizer=vectorizer, max_workers=settings.grpc_max_workers)
    bind_address = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(bind_address)

    server.start()
    logger.info("gRPC server listening on %s", bind_address)

    def handle_stop(signum, frame):
        logger.info("Received signal %s, shutting down gRPC server...", signum)
        server.stop(grace=5)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    server.wait_for_termination()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
