"""文本嵌入器 — 基于 TF-IDF + 截断 SVD 的轻量文本嵌入.

零重型依赖（仅 numpy + 标准库），与原项目设计原则一致。
将论文文本（标题+摘要+关键词）映射为稠密向量，用于隐性配对的相似度计算。

流程:
    1. 分词 → 2. 构建 TF-IDF 矩阵 → 3. 截断 SVD 降维 → 4. L2 归一化
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np



_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "for", "of", "to", "in", "on", "with", "and", "or",
    "is", "are", "we", "our", "by", "from", "as", "at", "this", "that",
    "these", "those", "it", "its", "be", "been", "was", "were", "has",
    "have", "had", "will", "would", "could", "should", "may", "might",
    "can", "than", "then", "so", "if", "but", "not", "no", "which", "who",
    "when", "where", "what", "how", "why", "all", "both", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "very",
    "just", "also", "into", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again", "further",
    "here", "there", "about", "against", "between", "them", "they", "their",
    "i", "you", "he", "she", "his", "her", "us", "him",
})


def tokenize(text: str, ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
    """分词 + n-gram 生成."""
    
    words = re.findall(r"[a-z]{2,}", text.lower())
    
    words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    tokens: List[str] = []
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        if n == 1:
            tokens.extend(words)
        else:
            for i in range(len(words) - n + 1):
                tokens.append(" ".join(words[i:i + n]))
    return tokens


class TextEmbedder:
    """文本嵌入器.

    使用 TF-IDF + 截断 SVD 实现轻量文本嵌入。
    支持增量更新（新文本可加入词汇表）。

    与原项目零重型依赖原则一致：仅使用 numpy。
    """

    def __init__(
        self,
        max_features: int = 2000,
        ngram_range: Tuple[int, int] = (1, 2),
        target_dim: int = 128,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.target_dim = target_dim

        
        self._vocab: Dict[str, int] = {}
        
        self._idf: Optional[np.ndarray] = None
        
        self._df: Dict[str, int] = defaultdict(int)
        
        self._n_docs: int = 0
        
        self._svd_components: Optional[np.ndarray] = None  # (target_dim, n_features)
        self._svd_mean: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._idf is not None and len(self._vocab) > 0

    def _build_vocab(self, texts: List[str]) -> None:
        """从文本列表构建词汇表."""
        
        df: Dict[str, int] = defaultdict(int)
        for text in texts:
            tokens = set(tokenize(text, self.ngram_range))
            for tok in tokens:
                df[tok] += 1

        
        sorted_tokens = sorted(df.items(), key=lambda x: -x[1])
        
        n = len(texts)
        min_df = max(1, n // 100)
        
        
        
        max_df = n if n < 20 else n * 0.98
        sorted_tokens = [
            (t, c) for t, c in sorted_tokens
            if c >= min_df and c <= max_df
        ][:self.max_features]

        self._vocab = {t: i for i, (t, _) in enumerate(sorted_tokens)}
        self._df = defaultdict(int, {t: c for t, c in sorted_tokens})
        self._n_docs = n

        
        self._idf = np.zeros(len(self._vocab))
        for t, i in self._vocab.items():
            df_t = self._df.get(t, 1)
            self._idf[i] = math.log((1 + n) / (1 + df_t)) + 1.0

    def _text_to_tfidf(self, text: str) -> np.ndarray:
        """将文本转为 TF-IDF 向量."""
        if not self._vocab or self._idf is None:
            return np.zeros(0)

        tokens = tokenize(text, self.ngram_range)
        tf = Counter(tokens)

        vec = np.zeros(len(self._vocab))
        for token, count in tf.items():
            idx = self._vocab.get(token)
            if idx is not None:
                vec[idx] = count * self._idf[idx]

        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _fit_svd(self, tfidf_matrix: np.ndarray) -> None:
        """拟合截断 SVD（使用随机化 SVD 近似）."""
        n_samples, n_features = tfidf_matrix.shape
        k = min(self.target_dim, n_features, n_samples)
        if k <= 0:
            self._svd_components = None
            return

        
        self._svd_mean = np.mean(tfidf_matrix, axis=0)
        centered = tfidf_matrix - self._svd_mean

        
        
        rng = np.random.RandomState(42)
        omega = rng.randn(n_features, min(k + 10, n_features))
        
        Y = centered @ omega  # (n_samples, k+10)
        
        Q, _ = np.linalg.qr(Y)  # Q: (n_samples, k+10)
        
        B = Q.T @ centered  # (k+10, n_features)
        try:
            U_b, S_b, Vt_b = np.linalg.svd(B, full_matrices=False)
        except np.linalg.LinAlgError:
            self._svd_components = None
            return

        
        self._svd_components = Vt_b[:k]  # (k, n_features)
        self._singular_values = S_b[:k]

    def _apply_svd(self, tfidf_vec: np.ndarray) -> np.ndarray:
        """应用 SVD 降维."""
        if self._svd_components is None:
            return tfidf_vec
        if self._svd_mean is not None:
            tfidf_vec = tfidf_vec - self._svd_mean
        reduced = self._svd_components @ tfidf_vec  # (k,)
        
        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced /= norm
        return reduced

    def fit(self, texts: List[str]) -> "TextEmbedder":
        """拟合嵌入器."""
        if not texts:
            return self
        self._build_vocab(texts)

        
        matrix = np.array([self._text_to_tfidf(t) for t in texts])
        if matrix.size > 0:
            self._fit_svd(matrix)
        return self

    def embed(self, text: str) -> np.ndarray:
        """嵌入单条文本.

        如果嵌入器未拟合，使用 TF-IDF 向量（不降维）。
        """
        if not self.is_fitted:
            
            return self._fallback_embed(text)

        
        
        
        
        
        return self._text_to_tfidf(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入."""
        return np.array([self.embed(t) for t in texts])

    def _fallback_embed(self, text: str) -> np.ndarray:
        """未拟合时的回退嵌入（基于词哈希）."""
        tokens = tokenize(text, self.ngram_range)
        if not tokens:
            return np.zeros(self.target_dim)

        vec = np.zeros(self.target_dim)
        for token in tokens:
            h = hash(token) % self.target_dim
            vec[h] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def partial_fit(self, texts: List[str]) -> "TextEmbedder":
        """增量拟合：将新文本加入词汇表，重新计算 IDF.

        用于 7×24 连续运行时，随新论文流入逐步完善嵌入空间。
        """
        if not texts:
            return self

        
        if not self.is_fitted:
            return self.fit(texts)

        
        for text in texts:
            tokens = set(tokenize(text, self.ngram_range))
            for tok in tokens:
                self._df[tok] += 1
            self._n_docs += 1

        
        n = self._n_docs
        for t, i in self._vocab.items():
            df_t = self._df.get(t, 1)
            self._idf[i] = math.log((1 + n) / (1 + df_t)) + 1.0

        
        new_tokens = [t for t, c in self._df.items()
                      if t not in self._vocab and c >= max(1, n // 100)]
        if new_tokens and len(self._vocab) + len(new_tokens) <= self.max_features:
            for t in new_tokens:
                idx = len(self._vocab)
                self._vocab[t] = idx
                
                df_t = self._df.get(t, 1)
                new_idf = math.log((1 + n) / (1 + df_t)) + 1.0
                self._idf = np.append(self._idf, new_idf)

            
            self._svd_components = None
            self._svd_mean = None

        return self
