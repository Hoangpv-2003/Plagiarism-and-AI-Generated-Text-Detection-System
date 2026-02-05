"""
Flask API Server for Plagiarism Detection
LoRewritten to match notebook logic 100%
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import pickle
import os
import re
import time
import difflib
import unicodedata
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

# ===============================================
# HELPER FUNCTIONS (FROM NOTEBOOK)
# ===============================================

def split_sentences_vi(text: str):
    """Sentence splitter - matches notebook logic"""
    raw = (text or "").strip()
    if not raw:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[\.!?…])\s+|\n+", raw) if s.strip()]
    return sents


def build_sentence_units(sentences, max_merge=2):
    """Build sentence units with merging - matches notebook"""
    sents = [s.strip() for s in (sentences or []) if isinstance(s, str) and s.strip()]
    if not sents:
        return []
    max_merge = max(1, min(int(max_merge or 2), 5))
    
    units = []
    for i in range(len(sents)):
        for m in range(1, max_merge + 1):
            j = i + m
            if j > len(sents):
                break
            units.append({
                'start': i,
                'end': j - 1,
                'span': (i, j - 1),
                'size': m,
                'text': " ".join(sents[i:j]).strip(),
            })
    units.sort(key=lambda u: (u['end'] - u['start'] + 1, len(u['text'])))
    return units


def expand_query_context(query_chunks, idx, context_window=1):
    """Expand context around a query chunk"""
    if not query_chunks or idx is None or idx < 0 or idx >= len(query_chunks):
        return ""
    context_window = max(0, min(int(context_window or 1), 5))
    start = max(0, idx - context_window)
    end = min(len(query_chunks), idx + context_window + 1)
    return " ".join([query_chunks[i].get('text', '').strip() for i in range(start, end)]).strip()


def _tokenize_simple(text: str):
    """Simple tokenization"""
    text = (text or "").lower()
    toks = re.findall(r"[\w]+", text, flags=re.UNICODE)
    return [t for t in toks if len(t) > 1]


def lexical_overlap_ratio(text_a: str, text_b: str) -> float:
    """Jaccard overlap on tokens"""
    a = set(_tokenize_simple(text_a))
    b = set(_tokenize_simple(text_b))
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


# ===============================================
# PHOBERT PARAPHRASE VERIFIER (FROM NOTEBOOK)
# ===============================================

class PhoBERTParaphraseVerifier:
    """Binary paraphrase verifier - matches notebook implementation"""
    
    def __init__(self, model_dir: str, device=None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        self.torch = torch
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))
    
    def predict_proba(self, text_a: str, text_b: str, max_length: int = 256) -> float:
        inputs = self.tokenizer(
            text_a or "",
            text_b or "",
            truncation=True,
            max_length=int(max_length),
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits
            if logits.shape[-1] == 1:
                return self._sigmoid(float(logits[0, 0].detach().cpu().item()))
            probs = self.torch.softmax(logits, dim=-1)
            return float(probs[0, -1].detach().cpu().item())


_paraphrase_verifier = None

def get_paraphrase_verifier():
    """Get or initialize paraphrase verifier"""
    global _paraphrase_verifier
    if _paraphrase_verifier is not None:
        return _paraphrase_verifier
    
    BASE_DIR = Path(__file__).resolve().parent.parent

    env_dir = (os.environ.get('PHOBERT_PARAPHRASE_MODEL_DIR') or '').strip()
    candidate_dirs = []
    if env_dir:
        candidate_dirs.append(Path(env_dir))
    candidate_dirs.extend([
        BASE_DIR / 'model' / 'phobert_finetuned',
        BASE_DIR / 'notebooks' / 'phobert_finetuned',
    ])

    model_dir = next((p for p in candidate_dirs if p.is_dir()), None)
    if model_dir is None:
        print(
            "⚠️ PhoBERT verifier not found. Tried: "
            + ", ".join(str(p) for p in candidate_dirs)
        )
        return None
    
    try:
        _paraphrase_verifier = PhoBERTParaphraseVerifier(str(model_dir))
        print(f"✅ Loaded PhoBERT paraphrase verifier from: {model_dir}")
        return _paraphrase_verifier
    except Exception as e:
        print(f"⚠️ Failed to load PhoBERT verifier: {e}")
        return None


# ===============================================
# TEXT CHUNKER (FROM NOTEBOOK)
# ===============================================

class TextChunker:
    """Sentence chunker with short-chunk merging - matches notebook"""
    
    def __init__(self, min_chunk_words=3, min_chunk_chars=12, merge_short_to='previous'):
        self.chunk_type = "sentence"
        self.min_chunk_words = int(min_chunk_words)
        self.min_chunk_chars = int(min_chunk_chars)
        assert merge_short_to in ('previous', 'next')
        self.merge_short_to = merge_short_to
    
    def chunk_text(self, text, doc_id):
        """Chunk by sentence with merging"""
        sentences = split_sentences_vi((text or "").strip())
        sentences = [s.strip() for s in sentences if s and s.strip()]
        
        chunks = []
        for i, sentence in enumerate(sentences):
            chunk = {
                'chunk_id': f"{doc_id}_chunk_{i}",
                'doc_id': doc_id,
                'text': sentence,
                'position': i,
                'length': len(sentence.split())
            }
            chunks.append(chunk)
        
        # Merge short chunks
        merged = []
        i = 0
        n = len(chunks)
        while i < n:
            c = chunks[i]
            is_short = (c['length'] < self.min_chunk_words) or (len(c['text']) < self.min_chunk_chars)
            if not is_short:
                merged.append(c.copy())
                i += 1
                continue
            
            if self.merge_short_to == 'previous' and merged:
                prev = merged[-1]
                prev['text'] = prev['text'] + ' ' + c['text']
                prev['length'] = len(prev['text'].split())
                i += 1
                continue
            elif self.merge_short_to == 'next' and (i + 1) < n:
                chunks[i+1]['text'] = c['text'] + ' ' + chunks[i+1]['text']
                chunks[i+1]['length'] = len(chunks[i+1]['text'].split())
                i += 1
                continue
            else:
                merged.append(c.copy())
                i += 1
        
        # Reassign positions and IDs
        for pos, chunk in enumerate(merged):
            chunk['position'] = pos
            chunk['chunk_id'] = f"{doc_id}_chunk_{pos}"
        
        return merged


# ===============================================
# DOCUMENT SCORER (FROM NOTEBOOK)
# ===============================================

class DocumentScorer:
    """Document scorer - matches notebook exactly"""
    
    def __init__(self, corpus_chunks, faiss_chunk_ids=None, weights=None,
                 score_center=0.50, score_scale=10.0, count_norm=6,
                 max_sim_k=5, contig_norm=4, sim_gate_low=0.60, sim_gate_high=0.78):
        self.corpus_chunks = corpus_chunks
        self.weights = weights or {
            'doc_max': 0.15,
            'doc_mean': 0.10,
            'doc_count': 0.20,
            'doc_contiguous': 0.40,
            'doc_coverage': 0.10,
            'chunk_density': 0.03,
            'span_penalty': 0.00,
            'contiguous_bonus': 0.25,
        }
        self.score_center = score_center
        self.score_scale = score_scale
        self.count_norm = max(int(count_norm), 1)
        self.max_sim_k = max(int(max_sim_k), 1)
        self.contig_norm = max(int(contig_norm), 1)
        self.sim_gate_low = float(sim_gate_low)
        self.sim_gate_high = float(sim_gate_high)
        
        self.faiss_chunk_ids = list(faiss_chunk_ids) if faiss_chunk_ids is not None else None
        self.chunk_map = {chunk['chunk_id']: chunk for chunk in corpus_chunks}
        self.doc_chunks_map = {}
        for chunk in corpus_chunks:
            self.doc_chunks_map.setdefault(chunk['doc_id'], []).append(chunk)
    
    def _get_chunk_by_faiss_idx(self, chunk_idx: int):
        if 0 <= chunk_idx < len(self.corpus_chunks):
            return self.corpus_chunks[chunk_idx]
        if self.faiss_chunk_ids is not None and 0 <= chunk_idx < len(self.faiss_chunk_ids):
            chunk_id = self.faiss_chunk_ids[chunk_idx]
            return self.chunk_map.get(chunk_id)
        return None
    
    def calculate_doc_scores(self, top_k_results):
        """Calculate document scores - matches notebook exactly"""
        doc_similarities = {}
        for similarity, chunk_idx in top_k_results:
            chunk_idx = int(chunk_idx)
            chunk = self._get_chunk_by_faiss_idx(chunk_idx)
            if chunk is None:
                continue
            doc_id = chunk['doc_id']
            doc_similarities.setdefault(doc_id, []).append({
                'similarity': similarity,
                'chunk': chunk,
                'chunk_idx': chunk_idx
            })
        
        doc_scores = []
        for doc_id, chunk_sims in doc_similarities.items():
            similarities = [float(cs['similarity']) for cs in chunk_sims]
            match_count = len(similarities)
            sims_sorted = sorted(similarities, reverse=True)
            doc_abs_max = sims_sorted[0]
            k = min(self.max_sim_k, len(sims_sorted))
            doc_max = float(np.mean(sims_sorted[:k]))
            doc_mean = float(np.mean(similarities))
            total_doc_chunks = len(self.doc_chunks_map[doc_id])
            coverage_ratio = min(match_count / total_doc_chunks, 1.0)
            doc_count = min(match_count / self.count_norm, 1.0)
            
            positions = sorted(cs['chunk']['position'] for cs in chunk_sims)
            doc_contiguous, max_group_len = self._calculate_contiguous_score(positions)
            span = positions[-1] - positions[0] + 1 if len(positions) > 1 else 1
            span_ratio = min(span / total_doc_chunks, 1.0)
            chunk_density = min(coverage_ratio / max(span_ratio, 1e-6), 1.0)
            chunk_similarity_std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
            
            # Similarity strength gating
            similarity_strength = (doc_max - self.sim_gate_low) / (self.sim_gate_high - self.sim_gate_low)
            similarity_strength = float(np.clip(similarity_strength, 0.0, 1.0))
            eff_doc_count = doc_count * similarity_strength
            eff_doc_contiguous = doc_contiguous * similarity_strength
            eff_coverage_ratio = coverage_ratio * similarity_strength
            
            # Contiguous bonus
            contiguous_len = int(max_group_len)
            contiguous_bonus_value = 0.0
            if contiguous_len >= 3:
                contiguous_bonus_value = min((contiguous_len - 2) / 8.0, 1.0) * doc_max
            
            raw_score = (
                self.weights['doc_max'] * doc_max +
                self.weights['doc_mean'] * doc_mean +
                self.weights['doc_count'] * eff_doc_count +
                self.weights['doc_contiguous'] * eff_doc_contiguous +
                self.weights['doc_coverage'] * eff_coverage_ratio +
                self.weights['chunk_density'] * chunk_density -
                self.weights['span_penalty'] * (1.0 - span_ratio) +
                self.weights.get('contiguous_bonus', 0.0) * contiguous_bonus_value
            )
            
            logistic_input = self.score_scale * (raw_score - self.score_center)
            final_score = 1.0 / (1.0 + np.exp(-logistic_input))
            
            doc_scores.append({
                'doc_id': doc_id,
                'doc_max': doc_max,
                'doc_abs_max': doc_abs_max,
                'doc_mean': doc_mean,
                'doc_count': doc_count,
                'doc_contiguous': doc_contiguous,
                'final_score': final_score,
                'raw_score': raw_score,
                'chunk_similarity_std': chunk_similarity_std,
                'position_span_ratio': span_ratio,
                'chunk_density': chunk_density,
                'contiguous_len': max_group_len,
                'contiguous_bonus': contiguous_bonus_value,
                'chunks': chunk_sims,
                'num_chunks': match_count,
                'coverage_ratio': coverage_ratio,
                'similarity_strength': similarity_strength,
                'eff_doc_count': eff_doc_count,
                'eff_doc_contiguous': eff_doc_contiguous,
                'eff_coverage_ratio': eff_coverage_ratio,
            })
        
        doc_scores.sort(key=lambda x: x['final_score'], reverse=True)
        return doc_scores
    
    def _calculate_contiguous_score(self, positions):
        """Calculate contiguous score"""
        if len(positions) <= 1:
            return 0.0, len(positions)
        contiguous_groups = []
        current_group = [positions[0]]
        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] <= 2:
                current_group.append(positions[i])
            else:
                contiguous_groups.append(len(current_group))
                current_group = [positions[i]]
        contiguous_groups.append(len(current_group))
        max_contiguous = max(contiguous_groups)
        contiguous_base = min(max_contiguous / len(positions), 1.0)
        support = min(len(positions) / self.contig_norm, 1.0)
        score = contiguous_base * support
        return float(score), int(max_contiguous)


# ===============================================
# CONTEXT EXPANDER (FROM NOTEBOOK)
# ===============================================

class ContextExpander:
    """Context expander - matches notebook"""
    
    def __init__(self, corpus_chunks):
        self.corpus_chunks = corpus_chunks
        self.doc_chunks_map = {}
        for chunk in corpus_chunks:
            doc_id = chunk['doc_id']
            self.doc_chunks_map.setdefault(doc_id, []).append(chunk)
        for doc_id in self.doc_chunks_map:
            self.doc_chunks_map[doc_id].sort(key=lambda x: x['position'])
    
    def expand_chunk_context(self, chunk, context_window=1):
        """Expand chunk context"""
        if chunk is None:
            return ""
        doc_id = chunk['doc_id']
        position = chunk['position']
        doc_chunks = self.doc_chunks_map.get(doc_id, [])
        current_idx = next((idx for idx, c in enumerate(doc_chunks) if c['position'] == position), None)
        if current_idx is None:
            return chunk['text']
        start_idx = max(0, current_idx - int(context_window))
        end_idx = min(len(doc_chunks), current_idx + int(context_window) + 1)
        context_chunks = doc_chunks[start_idx:end_idx]
        return " ".join([c['text'] for c in context_chunks])


# ===============================================
# LABELING LOGIC (FROM NOTEBOOK)
# ===============================================

def compute_features_and_label_for_doc_scores(
    query_text,
    doc_scores,
    bi_encoder,
    chunk_embeddings_normalized,
    chunker,
    context_expander,
    doc_chunks_map,
    context_window=1,
    sentence_merge_n=2,
    top_n=10,
    paraphrase_verify=True,
    paraphrase_threshold=0.70,
    max_length_phobert=256,
):
    """
    Compute features and assign heuristic labels for top-N candidate docs.
    This matches the notebook logic EXACTLY.
    """
    
    # Thresholds (from notebook)
    TH_STRONG_COV = 0.02
    TH_STRONG_FINAL = 0.70
    TH_STRONG_DMAX = 0.85
    
    TH_MIX_MIN_STRONG = 2
    TH_MIX_COV_LOW = TH_STRONG_COV
    TH_MIX_COV_HIGH = 1.00
    TH_MIX_STRONG_COV_MIN = 0.10
    TH_MIX_STRONG_CONTIG_MIN = 2
    TH_MIX_STRONG_NUMCHUNKS_MIN = 2
    
    TH_WHOLEDOC_COV = 0.60
    TH_LEN_RATIO_WHOLEDOC = 0.90
    
    TH_COPY_LEX_FULL = 0.90
    TH_COPY_SPAN_FULL = 0.85
    TH_COPY_DMAX = 0.90
    
    TH_INSERT_LEX_PAIR = 0.85
    TH_INSERT_COV_MAX = 0.60
    TH_INSERT_DMAX = 0.92
    TH_INSERT_QCHUNK_SIM = 0.98
    TH_INSERT_CONTIG_LEN = 3
    
    TH_COPYLIKE_PAIR_LEX = 0.95
    TH_COPYLIKE_PAIR_COV_MIN = 0.20
    
    TH_STRUCT_SENT_ORDER = 0.98
    TH_STRUCT_SENT_MULTISET = 0.60
    
    TH_PARA_FULL_COV_MIN = 0.60
    TH_PARA_FULL_COV_SOFT = 0.45
    TH_PARA_FULL_FINAL_SOFT = 0.95
    TH_PARA_FULL_QCHUNK_SIM_SOFT = 0.88
    TH_PARA_FULL_CONTIG_MIN_ABS = 8
    TH_PARA_FULL_CONTIG_MIN_FRAC = 0.45
    TH_PARA_FULL_LEX_MAX = 0.85
    
    rows = []
    top_docs = doc_scores[:min(len(doc_scores), int(top_n))]
    local_merge_n = int(sentence_merge_n) if sentence_merge_n is not None else 2
    
    # Helper functions
    def _tokenize(text: str):
        text = (text or "").lower()
        toks = re.findall(r"[\w]+", text, flags=re.UNICODE)
        return [t for t in toks if len(t) > 1]
    
    def _order_sim_ratio(tokens_a, tokens_b):
        if not tokens_a or not tokens_b:
            return 0.0
        sm = difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False)
        return float(sm.ratio())
    
    def _bigram_jaccard(tokens_a, tokens_b):
        if len(tokens_a) < 2 or len(tokens_b) < 2:
            return 0.0
        a = set(zip(tokens_a, tokens_a[1:]))
        b = set(zip(tokens_b, tokens_b[1:]))
        inter = len(a & b)
        union = len(a | b)
        return float(inter) / float(union) if union else 0.0
    
    def _sent_list(text: str):
        raw = text or ""
        try:
            sents = split_sentences_vi(raw)
        except Exception:
            sents = []
        if not sents:
            sents = re.split(r"(?<=[\.!?…])\s+|\n+", raw)
        sents = [s.strip() for s in sents if isinstance(s, str) and s.strip()]
        return sents
    
    def _sent_order_sim(sent_a, sent_b):
        if not sent_a or not sent_b:
            return 0.0
        sm = difflib.SequenceMatcher(a=sent_a, b=sent_b, autojunk=False)
        return float(sm.ratio())
    
    def _norm_ws(text: str):
        raw = unicodedata.normalize("NFKC", (text or ""))
        toks = re.findall(r"[\w]+", raw.lower(), flags=re.UNICODE)
        return " ".join(toks).strip()
    
    def _longest_common_span_len(a: str, b: str) -> int:
        if not a or not b:
            return 0
        m = difflib.SequenceMatcher(None, a, b, autojunk=False)
        match = m.find_longest_match(0, len(a), 0, len(b))
        return int(match.size)
    
    def _sent_multiset_overlap_ratio(sent_a, sent_b):
        a = [_norm_ws(s) for s in (sent_a or []) if isinstance(s, str) and s.strip()]
        b = [_norm_ws(s) for s in (sent_b or []) if isinstance(s, str) and s.strip()]
        if not a or not b:
            return 0.0
        ca = Counter(a)
        cb = Counter(b)
        inter = sum((ca & cb).values())
        denom = max(1, max(sum(ca.values()), sum(cb.values())))
        return float(inter) / float(denom)
    
    def _len_ratio(tokens_a, tokens_b):
        if not tokens_a and not tokens_b:
            return 0.0
        return min(len(tokens_a), len(tokens_b)) / max(1, max(len(tokens_a), len(tokens_b)))
    
    def _get_doc_text_by_id(target_doc_id: str):
        doc_chunks = doc_chunks_map.get(target_doc_id)
        if doc_chunks:
            doc_chunks_sorted = sorted(doc_chunks, key=lambda c: int(c.get('position', 0)))
            doc_text = " ".join([c.get('text', '') for c in doc_chunks_sorted if c.get('text')])
            if doc_text:
                return doc_text
        return None
    
    def _normalize_emb(x: np.ndarray) -> np.ndarray:
        if x is None:
            return None
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return (x / norms).astype("float32")
    
    # Strong-source count for label 3
    strong_doc_ids = set()
    for d in top_docs:
        try:
            _final = float(d.get("final_score", 0.0))
            _dmax = float(d.get("doc_max", 0.0))
            _cov = float(d.get("coverage_ratio", 0.0))
            _contig = int(d.get("contiguous_len", 0) or 0)
            _nchunks = int(d.get("num_chunks", 0) or 0)
            is_strong_for_mix = (
                (_cov >= TH_MIX_STRONG_COV_MIN)
                and (_cov >= TH_STRONG_COV)
                and (_final >= TH_STRONG_FINAL or _dmax >= TH_STRONG_DMAX)
                and (_contig >= TH_MIX_STRONG_CONTIG_MIN or _nchunks >= TH_MIX_STRONG_NUMCHUNKS_MIN)
            )
            if is_strong_for_mix:
                strong_doc_ids.add(str(d.get("doc_id")))
        except Exception:
            pass
    strong_doc_count = len(strong_doc_ids)
    
    # Decide which docs to label
    docs_to_label = []
    if top_docs:
        if strong_doc_count >= TH_MIX_MIN_STRONG:
            docs_to_label = [d for d in top_docs if str(d.get("doc_id")) in strong_doc_ids]
            if not docs_to_label:
                docs_to_label = top_docs[:1]
        else:
            docs_to_label = top_docs[:1]
    
    # Precompute query chunks and embeddings
    query_chunks = chunker.chunk_text(query_text, doc_id="query_for_features")
    q_chunk_texts = [c["text"] for c in query_chunks]
    q_chunk_emb_norm = None
    if q_chunk_texts:
        q_emb = bi_encoder.encode(q_chunk_texts, convert_to_numpy=True, show_progress_bar=False)
        q_chunk_emb_norm = _normalize_emb(q_emb)
    
    # Precompute query sentence-units
    q_sents = split_sentences_vi(query_text)
    q_units = build_sentence_units(q_sents, max_merge=local_merge_n)
    q_unit_texts = [u["text"] for u in q_units]
    q_unit_emb_norm = None
    if q_unit_texts:
        q_u_emb = bi_encoder.encode(q_unit_texts, convert_to_numpy=True, show_progress_bar=False)
        q_unit_emb_norm = _normalize_emb(q_u_emb)
    
    # Optional paraphrase verifier
    verifier = None
    if paraphrase_verify:
        try:
            verifier = get_paraphrase_verifier()
        except Exception:
            verifier = None
    
    def _best_query_chunk_for_corpus_idx(corpus_idx: int):
        if q_chunk_emb_norm is None or corpus_idx is None or corpus_idx < 0:
            return -1, 0.0
        c_vec = chunk_embeddings_normalized[int(corpus_idx)].astype("float32")
        q_sims = np.dot(q_chunk_emb_norm, c_vec)
        q_best_idx = int(np.argmax(q_sims))
        return q_best_idx, float(q_sims[q_best_idx])
    
    def _compute_paraphrase_scores_for_doc(ds):
        """Bidirectional many-to-one paraphrase verification - matches notebook exactly"""
        if verifier is None:
            return None, None, 0
        
        # A side: FULL query
        a_sents = _sent_list(query_text)
        a_sents = [s for s in a_sents if s]
        if not a_sents:
            return None, None, 0
        
        # B side: FULL document
        doc_id_local = ds.get("doc_id")
        if doc_id_local is None:
            return None, None, 0
        doc_text_local = _get_doc_text_by_id(str(doc_id_local))
        if not doc_text_local or not isinstance(doc_text_local, str):
            return None, None, 0
        b_sents = _sent_list(doc_text_local)
        b_sents = [s for s in b_sents if s]
        if not b_sents:
            return None, None, 0
        
        # Safety cap
        if len(a_sents) > 200:
            a_sents = a_sents[:200]
        if len(b_sents) > 200:
            b_sents = b_sents[:200]
        
        try:
            a_emb = _normalize_emb(bi_encoder.encode(a_sents, convert_to_numpy=True, show_progress_bar=False))
            b_emb = _normalize_emb(bi_encoder.encode(b_sents, convert_to_numpy=True, show_progress_bar=False))
        except Exception:
            return None, None, 0
        
        if a_emb is None or b_emb is None or a_emb.size == 0 or b_emb.size == 0:
            return None, None, 0
        
        sims = np.dot(a_emb, b_emb.T)
        if sims.size == 0:
            return None, None, 0
        
        # Step 1: A -> best B
        a_best_b = np.argmax(sims, axis=1)
        scores_a_to_b = []
        best_prob = None
        best_lex = None
        checked = 0
        for ai, bj in enumerate(a_best_b.tolist()):
            a_txt = a_sents[int(ai)].strip()
            b_txt = b_sents[int(bj)].strip()
            if not a_txt or not b_txt:
                continue
            try:
                prob = float(verifier.predict_proba(a_txt, b_txt, max_length=max_length_phobert))
            except Exception:
                continue
            scores_a_to_b.append(prob)
            checked += 1
            if best_prob is None or prob > best_prob:
                best_prob = prob
                try:
                    best_lex = float(lexical_overlap_ratio(a_txt, b_txt))
                except Exception:
                    best_lex = None
        
        # Step 2: B -> best A
        b_best_a = np.argmax(sims, axis=0)
        scores_b_to_a = []
        for bj, ai in enumerate(b_best_a.tolist()):
            a_txt = a_sents[int(ai)].strip()
            b_txt = b_sents[int(bj)].strip()
            if not a_txt or not b_txt:
                continue
            try:
                prob = float(verifier.predict_proba(a_txt, b_txt, max_length=max_length_phobert))
            except Exception:
                continue
            scores_b_to_a.append(prob)
            checked += 1
            if best_prob is None or prob > best_prob:
                best_prob = prob
                try:
                    best_lex = float(lexical_overlap_ratio(a_txt, b_txt))
                except Exception:
                    best_lex = None
        
        if not scores_a_to_b or not scores_b_to_a:
            return None, best_lex, int(checked)
        
        final_score = (float(np.mean(scores_a_to_b)) + float(np.mean(scores_b_to_a))) / 2.0
        return float(final_score), best_lex, int(checked)
    
    # Main loop
    for ds in docs_to_label:
        doc_id = str(ds.get('doc_id'))
        final = float(ds.get('final_score', 0.0))
        doc_max = float(ds.get('doc_max', 0.0))
        doc_mean = float(ds.get('doc_mean', 0.0))
        coverage = float(ds.get('coverage_ratio', 0.0))
        contig_len = int(ds.get('contiguous_len', 0))
        num_chunks = int(ds.get('num_chunks', 0))
        
        top_chunk = None
        top_chunk_idx = -1
        try:
            top_chunk = ds['chunks'][0]['chunk']
            top_chunk_idx = int(ds['chunks'][0].get('chunk_idx', -1))
        except Exception:
            pass
        
        corpus_ctx = ""
        query_ctx = query_text
        best_query_chunk_sim = 0.0
        best_query_chunk_idx = -1
        if top_chunk is not None:
            corpus_ctx = context_expander.expand_chunk_context(top_chunk, context_window=int(context_window))
        if (top_chunk_idx >= 0) and (q_chunk_emb_norm is not None):
            best_query_chunk_idx, best_query_chunk_sim = _best_query_chunk_for_corpus_idx(top_chunk_idx)
            if 0 <= best_query_chunk_idx < len(query_chunks):
                query_ctx = expand_query_context(query_chunks, best_query_chunk_idx, context_window=int(context_window))
        
        # Sentence-unit similarity
        best_sent = 0.0
        exact_copy_sentence_unit = False
        try:
            b_sents = split_sentences_vi(corpus_ctx)
            b_units = build_sentence_units(b_sents, max_merge=local_merge_n)
            b_texts = [u["text"] for u in b_units]
            if q_unit_texts and b_texts:
                q_set = {t.strip() for t in q_unit_texts}
                for b_t in b_texts:
                    if b_t.strip() in q_set:
                        exact_copy_sentence_unit = True
                        break
            if exact_copy_sentence_unit:
                best_sent = 1.0
            elif q_unit_emb_norm is not None and b_texts:
                b_emb = _normalize_emb(bi_encoder.encode(b_texts, convert_to_numpy=True, show_progress_bar=False))
                sim = np.dot(q_unit_emb_norm, b_emb.T)
                sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0)
                best_sent = float(sim.max()) if sim.size else 0.0
        except Exception:
            pass
        
        # Lexical features
        lex_overlap_ctx = float(lexical_overlap_ratio(query_ctx, corpus_ctx))
        q_tokens_ctx = _tokenize(query_ctx)
        c_tokens_ctx = _tokenize(corpus_ctx)
        order_sim_ctx = _order_sim_ratio(q_tokens_ctx, c_tokens_ctx)
        bigram_overlap_ctx = _bigram_jaccard(q_tokens_ctx, c_tokens_ctx)
        
        # Sentence features
        q_sent_ctx = _sent_list(query_ctx)
        c_sent_ctx = _sent_list(corpus_ctx)
        sent_count_diff_ctx = abs(len(q_sent_ctx) - len(c_sent_ctx))
        sent_order_sim_ctx = _sent_order_sim(q_sent_ctx, c_sent_ctx)
        
        # Whole-doc features
        doc_text_exact = False
        doc_text = None
        lex_overlap_full = None
        tok_len_ratio_full = None
        order_sim_full = None
        sent_order_sim_full = None
        sent_count_diff_full = None
        sent_multiset_overlap_full = None
        copy_span_ratio_full = None
        
        try:
            doc_text = _get_doc_text_by_id(doc_id)
            if doc_text:
                doc_text_exact = (_norm_ws(query_text) == _norm_ws(doc_text))
                q_tok_full = _tokenize(query_text)
                c_tok_full = _tokenize(doc_text)
                tok_len_ratio_full = _len_ratio(q_tok_full, c_tok_full)
                order_sim_full = _order_sim_ratio(q_tok_full, c_tok_full)
                
                q_sent_full = _sent_list(query_text)
                c_sent_full = _sent_list(doc_text)
                sent_count_diff_full = abs(len(q_sent_full) - len(c_sent_full))
                sent_order_sim_full = _sent_order_sim(q_sent_full, c_sent_full)
                sent_multiset_overlap_full = _sent_multiset_overlap_ratio(q_sent_full, c_sent_full)
                
                norm_q_full = _norm_ws(query_text)
                norm_c_full = _norm_ws(doc_text)
                span = _longest_common_span_len(norm_q_full, norm_c_full)
                copy_span_ratio_full = float(span) / float(max(1, min(len(norm_q_full), len(norm_c_full))))
                
                lex_overlap_full = float(lexical_overlap_ratio(query_text, doc_text))
        except Exception:
            pass
        
        # Paraphrase scores (lazy computation)
        paraphrase_score = None
        paraphrase_bestpair_lex = None
        paraphrase_pairs_checked = 0
        
        # ==================
        # LABEL ASSIGNMENT (MATCHES NOTEBOOK EXACTLY)
        # ==================
        label = 0
        paraphrase_candidate = False
        
        # Label 3: Mix
        if strong_doc_count >= TH_MIX_MIN_STRONG and str(doc_id) in strong_doc_ids and coverage >= TH_MIX_COV_LOW and coverage <= TH_MIX_COV_HIGH:
            label = 3
        
        # Label 2: Structural change
        if label == 0:
            structural_change = False
            whole_copy_guard = False
            if doc_text_exact:
                whole_copy_guard = True
            elif doc_text and (lex_overlap_full is not None) and (copy_span_ratio_full is not None):
                whole_copy_guard = (
                    (coverage >= TH_WHOLEDOC_COV)
                    and (doc_max >= TH_COPY_DMAX or final >= 0.90)
                    and (lex_overlap_full >= TH_COPY_LEX_FULL)
                    and (copy_span_ratio_full >= TH_COPY_SPAN_FULL)
                )
            
            if whole_copy_guard:
                structural_change = False
            elif doc_text and tok_len_ratio_full is not None and sent_order_sim_full is not None and sent_count_diff_full is not None:
                structural_change = (
                    (coverage >= TH_WHOLEDOC_COV)
                    and (tok_len_ratio_full >= TH_LEN_RATIO_WHOLEDOC)
                    and ((sent_count_diff_full > 0) or (sent_order_sim_full < TH_STRUCT_SENT_ORDER))
                    and (
                        (sent_multiset_overlap_full is not None and sent_multiset_overlap_full >= TH_STRUCT_SENT_MULTISET)
                        or (order_sim_full is not None and order_sim_full >= 0.60)
                    )
                )
            else:
                structural_change = (
                    (coverage >= TH_WHOLEDOC_COV)
                    and (_len_ratio(q_tokens_ctx, c_tokens_ctx) >= TH_LEN_RATIO_WHOLEDOC)
                    and ((sent_count_diff_ctx > 0) or (sent_order_sim_ctx < TH_STRUCT_SENT_ORDER))
                    and ((lex_overlap_ctx >= 0.60) or (order_sim_ctx >= 0.60))
                )
            
            if structural_change and (doc_max >= 0.85 or final >= 0.80):
                label = 2
        
        # Label 1: Copy
        if label == 0:
            if doc_text_exact:
                label = 1
            else:
                is_whole_copy = False
                if doc_text and lex_overlap_full is not None and copy_span_ratio_full is not None:
                    is_whole_copy = (
                        (coverage >= TH_WHOLEDOC_COV)
                        and (doc_max >= TH_COPY_DMAX or final >= 0.90)
                        and (lex_overlap_full >= TH_COPY_LEX_FULL)
                        and (copy_span_ratio_full >= TH_COPY_SPAN_FULL)
                    )
                
                copy_like_pair = (
                    (lex_overlap_ctx >= TH_COPYLIKE_PAIR_LEX)
                    and (doc_max >= TH_COPY_DMAX)
                    and (coverage >= TH_COPYLIKE_PAIR_COV_MIN)
                )
                
                is_partial_insert = (
                    (coverage < TH_INSERT_COV_MAX)
                    and (
                        exact_copy_sentence_unit
                        or (lex_overlap_ctx >= TH_INSERT_LEX_PAIR)
                        or (
                            (doc_max >= TH_INSERT_DMAX)
                            and (best_query_chunk_sim >= TH_INSERT_QCHUNK_SIM)
                            and (coverage >= 0.10)
                            and (contig_len >= TH_INSERT_CONTIG_LEN)
                        )
                    )
                )
                
                near_verbatim_ctx = (
                    exact_copy_sentence_unit
                    or (
                        (doc_max >= 0.90 and coverage >= 0.10 and lex_overlap_ctx >= 0.85 and order_sim_ctx >= 0.80)
                        or (doc_max >= 0.93 and best_sent >= 0.93 and coverage >= 0.10)
                    )
                )
                
                if is_whole_copy or is_partial_insert or near_verbatim_ctx or copy_like_pair:
                    label = 1
        
        # Label 4: Paraphrase
        if label == 0:
            span_proxy = None
            try:
                nqc = _norm_ws(query_ctx)
                ncc = _norm_ws(corpus_ctx)
                span_proxy = float(_longest_common_span_len(nqc, ncc)) / float(max(1, min(len(nqc), len(ncc))))
            except Exception:
                span_proxy = None
            
            looks_like_copy = (
                exact_copy_sentence_unit
                or (lex_overlap_ctx >= TH_COPYLIKE_PAIR_LEX)
                or (lex_overlap_full is not None and lex_overlap_full >= TH_COPYLIKE_PAIR_LEX)
                or (best_sent >= 0.97)
                or (span_proxy is not None and span_proxy >= 0.85)
            )
            
            reformulation_ok = (order_sim_ctx < 0.80) or (bigram_overlap_ctx < 0.55)
            
            contig_need = max(int(TH_PARA_FULL_CONTIG_MIN_ABS), int(num_chunks * TH_PARA_FULL_CONTIG_MIN_FRAC))
            paraphrase_full_ok = (coverage >= TH_PARA_FULL_COV_MIN) or (
                (coverage >= TH_PARA_FULL_COV_SOFT)
                and (final >= TH_PARA_FULL_FINAL_SOFT)
                and (best_query_chunk_sim >= TH_PARA_FULL_QCHUNK_SIM_SOFT)
                and (contig_len >= contig_need)
            )
            
            should_try_paraphrase = (
                paraphrase_verify
                and (not looks_like_copy)
                and reformulation_ok
                and paraphrase_full_ok
                and (lex_overlap_ctx < TH_PARA_FULL_LEX_MAX)
            )
            
            if should_try_paraphrase:
                paraphrase_score, paraphrase_bestpair_lex, paraphrase_pairs_checked = _compute_paraphrase_scores_for_doc(ds)
                paraphrase_verified = (
                    (paraphrase_score is not None)
                    and (int(paraphrase_pairs_checked) >= 1)
                    and (float(paraphrase_score) >= float(paraphrase_threshold))
                )
                
                if paraphrase_verified:
                    paraphrase_candidate = True
                    label = 4
        
        rows.append({
            'doc_id': doc_id,
            'heuristic_label': int(label),
            'final_score': float(final),
            'doc_max': float(doc_max),
            'doc_mean': float(doc_mean),
            'coverage_ratio': float(coverage),
            'best_sentence_sim': float(best_sent),
            'best_query_chunk_sim': float(best_query_chunk_sim),
            'lexical_overlap_ctx': float(lex_overlap_ctx),
            'order_sim_ctx': float(order_sim_ctx),
            'bigram_overlap_ctx': float(bigram_overlap_ctx),
            'sent_order_sim_ctx': float(sent_order_sim_ctx),
            'sent_count_diff_ctx': int(sent_count_diff_ctx),
            'doc_text_exact': bool(doc_text_exact),
            'contiguous_len': int(contig_len),
            'num_chunks': int(num_chunks),
            'strong_doc_count': int(strong_doc_count),
            'paraphrase_candidate': bool(paraphrase_candidate),
            'paraphrase_score': paraphrase_score,
            'paraphrase_bestpair_lex': paraphrase_bestpair_lex,
            'paraphrase_pairs_checked': int(paraphrase_pairs_checked),
        })
    
    return rows


# ===============================================
# FLASK APP
# ===============================================

app = Flask(__name__)
CORS(app)

# Load models and data
print("="*60)
print("🚀 LOADING MODELS AND DATA...")
print("="*60)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# Load bi-encoder
model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
bi_encoder = SentenceTransformer(model_name)
print(f"✅ Loaded bi-encoder: {model_name}")

# Load corpus chunks
with open(DATA_DIR / 'corpus_chunks.pkl', 'rb') as f:
    corpus_chunks = pickle.load(f)
print(f"✅ Loaded corpus chunks: {len(corpus_chunks)} chunks")

# Load embeddings
chunk_embeddings_normalized = np.load(DATA_DIR / 'chunk_embeddings_normalized.npy', mmap_mode='r')
print(f"✅ Loaded embeddings: {chunk_embeddings_normalized.shape}")

# Load chunk metadata
with open(DATA_DIR / 'chunk_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
chunk_ids = metadata['chunk_ids']
print(f"✅ Loaded metadata: {len(chunk_ids)} chunk IDs")

# Load corpus document metadata (URL, title) from parquet files
corpus_doc_info = {}  # doc_id -> {url, title}
doc_index_to_hash = {}  # integer index -> hash ID
try:
    import glob
    corpus_files = glob.glob(str(DATA_DIR / 'corpus' / '*.parquet'))
    if corpus_files:
        import pandas as pd
        print(f"📚 Loading corpus metadata from {len(corpus_files)} parquet files...")
        for pf in corpus_files[:1]:  # Load first file for now
            print(f"   📄 File: {os.path.basename(pf)}")
            df = pd.read_parquet(pf)
            print(f"      Rows: {len(df)}, Columns: {list(df.columns)}")
            
            for idx, row in df.iterrows():
                # Extract hash ID from extra_metadata dict
                extra_meta = row.get('extra_metadata', {})
                if isinstance(extra_meta, dict) and 'id' in extra_meta:
                    hash_id = str(extra_meta['id'])
                else:
                    hash_id = f"doc_{idx}"
                
                # Map both integer index and hash ID
                doc_index_to_hash[str(idx)] = hash_id
                
                corpus_doc_info[hash_id] = {
                    'url': str(row.get('url', '')).strip() if pd.notna(row.get('url')) else '',
                    'title': str(row.get('title', '')).strip() if pd.notna(row.get('title')) else f"Document {hash_id}"
                }
                
                # Also map by integer index for direct access
                corpus_doc_info[str(idx)] = corpus_doc_info[hash_id]
            
            # Show sample
            print(f"      Loaded {len(df)} documents")
            print(f"      Sample hash IDs: {list(doc_index_to_hash.values())[:3]}")
            if corpus_doc_info:
                sample_hash = list(doc_index_to_hash.values())[0]
                sample_doc = corpus_doc_info[sample_hash]
                print(f"      First doc url: {sample_doc['url'][:80] if sample_doc['url'] else 'N/A'}")
                print(f"      First doc title: {sample_doc['title'][:80]}")
        
        print(f"✅ Loaded metadata for {len(doc_index_to_hash)} documents")
except Exception as e:
    print(f"⚠️ Could not load corpus metadata: {e}")
    import traceback
    traceback.print_exc()
    corpus_doc_info = {}
    doc_index_to_hash = {}

# Create maps
chunk_map = {chunk['chunk_id']: chunk for chunk in corpus_chunks}
doc_chunks_map = {}
for chunk in corpus_chunks:
    doc_chunks_map.setdefault(chunk['doc_id'], []).append(chunk)

# Initialize components
chunker = TextChunker()
doc_scorer = DocumentScorer(corpus_chunks, faiss_chunk_ids=chunk_ids)
context_expander = ContextExpander(corpus_chunks)

print("="*60)
print("✅ ALL MODELS LOADED!")
print("="*60)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Plagiarism Detection API is running'})


@app.route('/api/check-plagiarism', methods=['POST'])
def check_plagiarism():
    """Main plagiarism detection endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        query_text = data['text'].strip()
        if not query_text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        print(f"\n{'='*60}")
        print(f"📝 Processing query ({len(query_text)} chars)")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1: Chunk query
        query_chunks = chunker.chunk_text(query_text, doc_id="query")
        q_chunk_texts = [c['text'] for c in query_chunks]
        
        # Step 2: Encode query chunks
        q_emb = bi_encoder.encode(q_chunk_texts, show_progress_bar=False, convert_to_numpy=True)
        q_norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norms[q_norms < 1e-8] = 1.0
        q_emb_norm = (q_emb / q_norms).astype('float32')
        
        # Step 3: Search (dot product)
        similarity_matrix = np.dot(q_emb_norm, chunk_embeddings_normalized.T)
        corpus_scores = np.max(similarity_matrix, axis=0)
        top_k = 100
        top_k_indices = np.argsort(corpus_scores)[::-1][:top_k]
        bi_results = [(float(corpus_scores[idx]), int(idx)) for idx in top_k_indices]
        
        # Step 4: Document scoring
        doc_scores = doc_scorer.calculate_doc_scores(bi_results)
        
        # Step 5: Labeling
        labeled_docs = compute_features_and_label_for_doc_scores(
            query_text,
            doc_scores,
            bi_encoder,
            chunk_embeddings_normalized,
            chunker,
            context_expander,
            doc_chunks_map,
            context_window=1,
            sentence_merge_n=2,
            top_n=10,
            paraphrase_verify=True,
            paraphrase_threshold=0.70,
        )
        
        detection_time = time.time() - start_time
        
        # Get top label
        top_label = 0
        confidence_value = 0.0
        if labeled_docs:
            top_row = labeled_docs[0]
            top_label = int(top_row.get('heuristic_label', 0))
            if top_label == 4:
                confidence_value = float(top_row.get('paraphrase_score') or top_row.get('final_score', 0.0))
            else:
                confidence_value = float(top_row.get('final_score', 0.0))
        
        # Create sentence_analysis for highlighting
        sentence_analysis = []
        try:
            if query_chunks and len(query_chunks) > 0:
                # Get max similarity for each query chunk
                chunk_max_sims = np.max(similarity_matrix, axis=1)
                # Get best matching corpus chunk index for each query chunk
                chunk_best_matches = np.argmax(similarity_matrix, axis=1)
                
                print(f"📊 Creating sentence analysis for {len(query_chunks)} chunks")
                
                for i, chunk in enumerate(query_chunks):
                    chunk_text = chunk['text']
                    chunk_sim = float(chunk_max_sims[i])
                    best_corpus_idx = int(chunk_best_matches[i])
                    
                    # Determine if suspicious based on similarity threshold
                    is_suspicious = False
                    confidence = chunk_sim
                    
                    # High similarity chunks are suspicious
                    if chunk_sim >= 0.70:
                        is_suspicious = True
                    elif chunk_sim >= 0.60 and top_label in (1, 2, 3, 4):
                        is_suspicious = True
                    
                    # Get source document info for suspicious chunks
                    source_url = None
                    source_title = None
                    source_doc_id = None
                    
                    if is_suspicious and 0 <= best_corpus_idx < len(corpus_chunks):
                        best_chunk = corpus_chunks[best_corpus_idx]
                        source_doc_id = best_chunk.get('doc_id')
                        
                        print(f"  🔍 Chunk {i}: best_corpus_idx={best_corpus_idx}, source_doc_id={source_doc_id} (type: {type(source_doc_id)})")
                        
                        # Get URL and title from corpus_doc_info
                        if source_doc_id and source_doc_id in corpus_doc_info:
                            doc_info = corpus_doc_info[source_doc_id]
                            source_url = doc_info.get('url', '').strip()
                            source_title = doc_info.get('title', '').strip()
                            if not source_title:
                                source_title = f"Document {source_doc_id}"
                            print(f"  ✅ Found source: doc_id={source_doc_id}, url={source_url[:50]}..., title={source_title[:50]}")
                        else:
                            # Try converting doc_id to different types
                            source_doc_id_str = str(source_doc_id) if source_doc_id is not None else None
                            source_doc_id_int = int(source_doc_id) if source_doc_id is not None and str(source_doc_id).isdigit() else None
                            
                            print(f"  ⚠️ No metadata for doc_id={source_doc_id} (in corpus_doc_info: {source_doc_id in corpus_doc_info if source_doc_id else 'N/A'})")
                            print(f"     Tried str={source_doc_id_str} (exists: {source_doc_id_str in corpus_doc_info if source_doc_id_str else False})")
                            if source_doc_id_int is not None:
                                print(f"     Tried int={source_doc_id_int} (exists: {source_doc_id_int in corpus_doc_info})")
                            print(f"     Available keys sample: {list(corpus_doc_info.keys())[:5]}")

                    
                    sentence_analysis.append({
                        'sentence': chunk_text,
                        'is_suspicious': bool(is_suspicious),
                        'is_plagiarized': bool(is_suspicious),
                        'confidence': round(float(confidence), 4),
                        'similarity': round(float(chunk_sim), 4),
                        'word_count': len(chunk_text.split()),
                        'source_doc_id': source_doc_id if is_suspicious else None,
                        'source_url': source_url if (is_suspicious and source_url) else None,
                        'source_title': source_title if is_suspicious else None,
                    })
                
                print(f"✅ Created {len(sentence_analysis)} sentence analysis entries")
                
                # Debug: Show sample suspicious entries
                suspicious = [s for s in sentence_analysis if s.get('is_suspicious')]
                if suspicious:
                    print(f"   📌 Found {len(suspicious)} suspicious chunks")
                    print(f"   📌 First suspicious chunk:")
                    sample = suspicious[0]
                    print(f"      - sentence: {sample['sentence'][:50]}...")
                    print(f"      - source_doc_id: {sample.get('source_doc_id')}")
                    print(f"      - source_url: {sample.get('source_url', 'None')[:80] if sample.get('source_url') else 'None'}")
                    print(f"      - source_title: {sample.get('source_title', 'None')[:80] if sample.get('source_title') else 'None'}")
                else:
                    print(f"   ⚠️ No suspicious chunks found")
        except Exception as e:
            print(f"⚠️ Error creating sentence_analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # Map label to display
        label_map = {
            0: {"result_text": "Không đạo văn", "plagiarism_type": None},
            1: {"result_text": "Đạo văn", "plagiarism_type": "Sao chép"},
            2: {"result_text": "Đạo văn", "plagiarism_type": "Thay đổi cấu trúc văn bản"},
            3: {"result_text": "Đạo văn", "plagiarism_type": "Kết hợp nhiều nguồn"},
            4: {"result_text": "Diễn đạt lại", "plagiarism_type": None},
        }
        display_info = label_map.get(top_label, label_map[0])
        
        response = {
            'label': int(top_label),
            'heuristic_label': int(top_label),
            'result_text': display_info['result_text'],
            'plagiarism_type': display_info['plagiarism_type'],
            'is_plagiarism': bool(top_label in (1, 2, 3)),
            'confidence': round(float(confidence_value), 6),
            'text': query_text,  # Add original text for frontend display
            'sentence_analysis': sentence_analysis,  # Add sentence analysis for highlighting
            'labeled_candidates': labeled_docs,
            'stats': {
                'query_words': len(query_text.split()),
                'query_chunks': len(query_chunks),
                'corpus_matches': len(doc_scores),  # Number of documents found
                'detection_time': round(detection_time, 3),
                'total_time': round(detection_time, 3),  # Alias for compatibility
            }
        }
        
        print(f"✅ Detection completed in {detection_time:.3f}s")
        print(f"   Label: {top_label}, Confidence: {confidence_value:.4f}")
        
        # Debug: Print response structure
        print(f"📤 Sending response:")
        print(f"   - sentence_analysis entries: {len(response['sentence_analysis'])}")
        print(f"   - labeled_candidates entries: {len(response['labeled_candidates'])}")
        suspicious_with_url = [s for s in response['sentence_analysis'] if s.get('is_suspicious') and s.get('source_url')]
        print(f"   - suspicious with URL: {len(suspicious_with_url)}")
        if suspicious_with_url:
            print(f"   - Sample URL: {suspicious_with_url[0].get('source_url')[:80]}")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER")
    print("="*60)
    print("   Server: http://localhost:5000")
    print("   API: http://localhost:5000/api/check-plagiarism")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
