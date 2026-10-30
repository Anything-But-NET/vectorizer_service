import logging

import grpc
import numpy as np
from torch import Tensor

from vectorizer_service.grpc_gen import vectorizer_pb2, vectorizer_pb2_grpc
from vectorizer_service.services.transform_service import (
    FlatNodeData,
    VectorizationService,
)

logger = logging.getLogger(__name__)


def _encode_vector(vec: Tensor | None) -> bytes:
    if vec is None:
        return b""
    return np.asarray(vec.cpu().numpy().astype(np.float32), dtype=np.float32).tobytes()


class MarkdownVectorizerServicer(vectorizer_pb2_grpc.MarkdownVectorizerServicer):
    def __init__(self, vectorizer: VectorizationService | None = None):
        self.vectorizer = vectorizer or VectorizationService()

    def VectorizeQuery(
        self,
        request: vectorizer_pb2.QueryRequest,
        context: grpc.ServicerContext,
    ) -> vectorizer_pb2.QueryResponse:
        try:
            vec = self.vectorizer.vectorize_query(request.query)
            return vectorizer_pb2.QueryResponse(embedding=_encode_vector(vec))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            logger.exception("Error in VectorizeQuery")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def VectorizeTree(
        self,
        request: vectorizer_pb2.TreeRequest,
        context: grpc.ServicerContext,
    ) -> vectorizer_pb2.TreeResponse:
        try:
            nodes = [
                FlatNodeData(
                    name=node.name,
                    level=node.level,
                    content=node.content,
                    first_edge_idx=node.first_edge_idx,
                    edge_count=node.edge_count,
                )
                for node in request.nodes
            ]
            vectorized_nodes = self.vectorizer.vectorize_tree(
                nodes=nodes,
                child_indices=list(request.child_indices),
            )

            output_nodes = [
                vectorizer_pb2.FlatNodeOutput(
                    embedding=_encode_vector(v_node.embedding),
                    content=[
                        vectorizer_pb2.VectorizedChunk(
                            text=chunk_text,
                            embedding=_encode_vector(chunk_vec),
                        )
                        for chunk_text, chunk_vec in v_node.content
                    ],
                )
                for v_node in vectorized_nodes
            ]
            return vectorizer_pb2.TreeResponse(nodes=output_nodes)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            logger.exception("Error in VectorizeTree")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
