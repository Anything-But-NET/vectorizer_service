from collections.abc import Sequence
from dataclasses import dataclass, field
import re

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from vectorizer_service.config import get_settings


@dataclass
class FlatNodeData:
    name: str
    level: int
    content: str = ""
    first_edge_idx: int = 0
    edge_count: int = 0


@dataclass
class VectorizedNodeData:
    embedding: torch.Tensor | None = None
    content: list[tuple[str, torch.Tensor]] = field(default_factory=list)


class VectorizationService:
    SENTENCE_SPLIT_PATTERN = re.compile(
        r"([^\n]*\n(?:```[\s\S]*?```|~~~[\s\S]*?~~~))|(?<=[.!?])\s+|\n+"
    )
    SUBTOPIC_PREFIX = "**"
    SUBTOPIC_MIN_LEN = 25

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().embedding_model_name
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    @staticmethod
    def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
        return F.normalize(vec, p=2, dim=-1)

    @staticmethod
    def _merge_vectors(
        vectors: list[torch.Tensor], temperature: float = 0.1
    ) -> torch.Tensor:
        matrix = torch.stack(vectors)
    
        normalized = F.normalize(matrix, p=2, dim=1)
    
        attention_matrix = torch.matmul(normalized, normalized.T)
    
        scores = attention_matrix.mean(dim=1)
    
        weights = F.softmax(scores / temperature, dim=0)
    
        parent_vector = weights @ normalized
    
        return F.normalize(parent_vector, p=2, dim=0)

    def _split_oversized_sentences(
        self,
        sentences: list[str],
        max_chunk_size: int,
        chunk_overlap: int,
    ) -> list[tuple[str, int]]:
        token_ids_list = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        step = max(1, max_chunk_size - max(0, min(chunk_overlap, max_chunk_size - 1)))

        result: list[tuple[str, int]] = []
        for sentence, token_ids in zip(sentences, token_ids_list):
            s_len = len(token_ids)
            if s_len <= max_chunk_size:
                result.append((sentence, s_len))
                continue

            lines = [line.strip() for line in sentence.splitlines() if line.strip()]
            if len(lines) > 1:
                result.extend(
                    self._split_oversized_sentences(lines, max_chunk_size, chunk_overlap)
                )
                continue

            for start_idx in range(0, s_len, step):
                slice_ids = token_ids[start_idx : start_idx + max_chunk_size]
                slice_text = self.tokenizer.decode(slice_ids, skip_special_tokens=True).strip()
                if slice_text:
                    result.append((slice_text, len(slice_ids)))
                if start_idx + max_chunk_size >= s_len:
                    break

        return result

    def _micro_split_chunks(
        self,
        text: str,
        max_chunk_size: int = 100,
        chunk_overlap: int = 0,
    ) -> list[tuple[str, int, int]]:
        res: list[tuple[str, int, int]] = []

        raw_sentences = [
            s.strip()
            for s in self.SENTENCE_SPLIT_PATTERN.split(text)
            if s and s.strip()
        ]
        if not raw_sentences:
            return res

        sentences_with_lens = self._split_oversized_sentences(
            raw_sentences,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

        current_len = 0
        current_sentences: list[tuple[str, int]] = []
        start_token_idx = 1

        for sentence, s_len in sentences_with_lens:
            is_subtopic = (
                sentence.startswith(self.SUBTOPIC_PREFIX)
                and current_len >= self.SUBTOPIC_MIN_LEN
            )
            if not is_subtopic and current_len + s_len <= max_chunk_size:
                current_sentences.append((sentence, s_len))
                current_len += s_len
            else:
                if current_sentences:
                    chunk_text = "\n".join(s for s, _ in current_sentences)
                    res.append((chunk_text, start_token_idx, start_token_idx + current_len))

                while (
                    is_subtopic
                    or current_len > chunk_overlap
                    or current_len + s_len > max_chunk_size
                ) and current_sentences:
                    _, popped_len = current_sentences.pop(0)
                    current_len -= popped_len
                    start_token_idx += popped_len
                    if is_subtopic and current_len == 0:
                        break

                current_len += s_len
                current_sentences.append((sentence, s_len))

        if current_sentences:
            chunk_text = "\n".join(s for s, _ in current_sentences)
            res.append((chunk_text, start_token_idx, start_token_idx + current_len))

        return res

    def _get_embedding(
        self,
        data: str,
        macro_chunk_size: int = 7500,
        macro_chunk_overlap: int = 500,
    ) -> list[tuple[str, torch.Tensor]]:
        if not data.strip():
            return []

        macro_chunks = self._micro_split_chunks(
            data,
            max_chunk_size=macro_chunk_size,
            chunk_overlap=macro_chunk_overlap,
        )

        res: list[tuple[str, torch.Tensor]] = []
        prev_macro_end = 1

        for macro_idx, (macro_text, macro_start, macro_end) in enumerate(macro_chunks):
            if not macro_text.strip():
                continue

            inputs = self.tokenizer(macro_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state

            seq_len = embeddings.shape[1]

            overlap_offset = (prev_macro_end - macro_start) if macro_idx > 0 else 0
            min_micro_start = 1 + max(0, overlap_offset)

            micro_chunks = [
                (m_text, m_start, m_end)
                for m_text, m_start, m_end in self._micro_split_chunks(macro_text)
                if m_text.strip() and m_start >= min_micro_start
            ]
            if not micro_chunks:
                prev_macro_end = macro_end
                continue

            batch_inputs = self.tokenizer(
                [m_text for m_text, _, _ in micro_chunks],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                batch_cls = self.model(**batch_inputs).last_hidden_state[:, 0]

            for (micro_text, micro_start, micro_end), chunk_cls in zip(micro_chunks, batch_cls):
                s_idx = min(max(micro_start, 1), seq_len - 1)
                e_idx = min(max(micro_end, s_idx + 1), seq_len)

                late_span_vec = self._l2_normalize(
                    embeddings[0, s_idx:e_idx].mean(dim=0)
                )
                vec = self._l2_normalize(self._l2_normalize(chunk_cls) + late_span_vec)
                res.append((micro_text, vec))

            prev_macro_end = macro_end

        return res

    def vectorize_query(self, query: str) -> torch.Tensor:
        chunks = self._get_embedding(query)
        if not chunks:
            raise ValueError("Query cannot be empty")
        if len(chunks) == 1:
            return chunks[0][1]
        return self._merge_vectors([vec for _, vec in chunks])

    def vectorize_tree(
        self,
        nodes: Sequence[FlatNodeData],
        child_indices: Sequence[int],
    ) -> list[VectorizedNodeData]:
        n = len(nodes)
        if n == 0:
            return []

        results: list[VectorizedNodeData | None] = [None] * n
        visiting: set[int] = set()

        def process_node(idx: int) -> VectorizedNodeData:
            if results[idx] is not None:
                return results[idx]  # type: ignore[return-value]
            if idx in visiting:
                raise ValueError(f"Cycle detected in tree at node index {idx}")

            visiting.add(idx)
            node = nodes[idx]

            raw_content = node.content.strip()
            if raw_content and node.level > 0:
                raw_content = f"{'#' * node.level} {node.name}\n{raw_content}"
            content_chunks = self._get_embedding(raw_content)

            content_vectors = [embedding for _, embedding in content_chunks]
            all_vectors: list[torch.Tensor] = []

            if node.level > 0 and node.name.strip():
                name_embeddings = self._get_embedding(node.name)
                if name_embeddings:
                    all_vectors.append(name_embeddings[0][1])

            if content_vectors:
                all_vectors.append(self._merge_vectors(content_vectors))

            start = max(0, node.first_edge_idx)
            end = min(len(child_indices), start + max(0, node.edge_count))
            for edge_pos in range(start, end):
                child_idx = child_indices[edge_pos]
                if 0 <= child_idx < n:
                    child_res = process_node(child_idx)
                    if child_res.embedding is not None:
                        all_vectors.append(child_res.embedding)

            merged = self._merge_vectors(all_vectors) if all_vectors else None
            visiting.remove(idx)

            res = VectorizedNodeData(embedding=merged, content=content_chunks)
            results[idx] = res
            return res

        for i in range(n):
            if results[i] is None:
                process_node(i)

        return [r for r in results if r is not None]
