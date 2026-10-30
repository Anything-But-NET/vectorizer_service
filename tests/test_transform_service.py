from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest
import torch

from vectorizer_service.grpc_gen import vectorizer_pb2, vectorizer_pb2_grpc
from vectorizer_service.main import create_server
from vectorizer_service.servicer import MarkdownVectorizerServicer, _encode_vector
from vectorizer_service.services.transform_service import (
    FlatNodeData,
    VectorizationService,
)


def test_l2_normalize():
    zero_vec = torch.tensor([0.0, 0.0, 0.0])
    norm_zero = VectorizationService._l2_normalize(zero_vec)
    torch.testing.assert_close(zero_vec, norm_zero)

    vec = torch.tensor([3.0, 4.0])
    norm_vec = VectorizationService._l2_normalize(vec)
    torch.testing.assert_close(torch.linalg.norm(norm_vec), torch.tensor(1.0))
    torch.testing.assert_close(norm_vec, torch.tensor([0.6, 0.8]))


def test_merge_vectors():
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0])
    merged = VectorizationService._merge_vectors([v1, v2], temperature=0.1)
    assert merged.shape == (3,)
    torch.testing.assert_close(torch.linalg.norm(merged), torch.tensor(1.0))


def test_encode_vector():
    assert _encode_vector(None) == b""
    vec = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    encoded = _encode_vector(vec)
    decoded = np.frombuffer(encoded, dtype=np.float32)
    np.testing.assert_allclose(decoded, [1.0, 2.0, 3.0])


@pytest.fixture
def mock_vectorizer():
    with (
        patch("vectorizer_service.services.transform_service.AutoTokenizer") as mock_tok_cls,
        patch("vectorizer_service.services.transform_service.AutoModel") as mock_model_cls,
    ):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        def fake_tokenizer(inputs, add_special_tokens=False, return_tensors=None, **kwargs):
            if isinstance(inputs, list):
                if return_tensors == "pt":
                    mock_res = MagicMock()
                    mock_res.to.return_value = {
                        "input_ids": torch.ones((len(inputs), 5), dtype=torch.long),
                        "attention_mask": torch.ones((len(inputs), 5), dtype=torch.long),
                    }
                    return mock_res
                return {"input_ids": [[1, 2, 3] for _ in inputs]}
            elif isinstance(inputs, str):
                if return_tensors == "pt":
                    return {
                        "input_ids": torch.ones((1, 8), dtype=torch.long),
                        "attention_mask": torch.ones((1, 8), dtype=torch.long),
                    }
                return {"input_ids": [1, 2, 3]}
            return {"input_ids": []}

        mock_tokenizer.side_effect = fake_tokenizer
        mock_tokenizer.decode.return_value = "decoded sentence chunk"

        def fake_forward(**inputs):
            res = MagicMock()
            batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
            seq_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 8
            res.last_hidden_state = torch.ones((batch_size, seq_len, 4))
            return res

        mock_model.side_effect = fake_forward

        service = VectorizationService(model_name="dummy/model")
        return service


def test_split_oversized_sentences(mock_vectorizer):
    sentences = ["short", "multiline\nsentence\nthat\nis\nlong", "long sentence without breaks"]

    mock_vectorizer.tokenizer.side_effect = lambda s, add_special_tokens=False: {
        "input_ids": [[1, 2] if s_item == "short" else [1] * 20 for s_item in s]
    }
    mock_vectorizer.tokenizer.decode.return_value = "sliced text"

    res = mock_vectorizer._split_oversized_sentences(
        sentences,
        max_chunk_size=5,
        chunk_overlap=1,
    )
    assert len(res) >= 3
    assert res[0] == ("short", 2)


def test_micro_split_chunks_subtopics_and_overflow(mock_vectorizer):
    assert mock_vectorizer._micro_split_chunks("") == []
    assert mock_vectorizer._micro_split_chunks("   \n\t  ") == []

    mock_vectorizer.tokenizer.side_effect = lambda s, add_special_tokens=False: {
        "input_ids": [[1] * 10 for _ in s]
    }
    text = "First sentence. Second sentence. ** Subtopic header that is quite long enough."
    chunks = mock_vectorizer._micro_split_chunks(text, max_chunk_size=15, chunk_overlap=2)
    assert len(chunks) > 0


def test_get_embedding_empty_and_valid(mock_vectorizer):
    assert mock_vectorizer._get_embedding("") == []
    assert mock_vectorizer._get_embedding("   ") == []

    embeddings = mock_vectorizer._get_embedding("This is sample text for vectorization.")
    assert len(embeddings) > 0
    text, vec = embeddings[0]
    assert isinstance(text, str)
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (4,)


def test_vectorize_query(mock_vectorizer):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        mock_vectorizer.vectorize_query("")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        mock_vectorizer.vectorize_query("   ")

    with patch.object(mock_vectorizer, "_get_embedding") as mock_get_emb:
        v1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mock_get_emb.return_value = [("query text", v1)]
        res = mock_vectorizer.vectorize_query("query text")
        torch.testing.assert_close(res, v1)

        v2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
        mock_get_emb.return_value = [("q1", v1), ("q2", v2)]
        res_multi = mock_vectorizer.vectorize_query("long query")
        assert res_multi.shape == (4,)


def test_vectorize_tree(mock_vectorizer):
    assert mock_vectorizer.vectorize_tree([], []) == []

    nodes = [
        FlatNodeData(name="Root", level=0, content="", first_edge_idx=0, edge_count=1),
        FlatNodeData(name="Intro", level=1, content="Some introductory text.", first_edge_idx=1, edge_count=1),
        FlatNodeData(name="Empty Child", level=2, content="", first_edge_idx=0, edge_count=0),
    ]
    child_indices = [1, 2]

    fake_vec = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    with patch.object(mock_vectorizer, "_get_embedding", return_value=[("chunk", fake_vec)]):
        vec_nodes = mock_vectorizer.vectorize_tree(nodes, child_indices)
        assert len(vec_nodes) == 3
        assert vec_nodes[0].embedding is not None
        assert vec_nodes[1].embedding is not None
        assert len(vec_nodes[1].content) == 1
        assert vec_nodes[2].embedding is not None


def test_vectorize_tree_cycle_detection(mock_vectorizer):
    nodes = [
        FlatNodeData(name="Node 0", level=1, content="A", first_edge_idx=0, edge_count=1),
        FlatNodeData(name="Node 1", level=1, content="B", first_edge_idx=1, edge_count=1),
    ]
    child_indices = [1, 0]

    with pytest.raises(ValueError, match="Cycle detected"):
        mock_vectorizer.vectorize_tree(nodes, child_indices)
