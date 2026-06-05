import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class CacheConfig:
    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    tier0_size: int = 256
    tier1_size: int = 2048
    tier1_compression: int = 4
    tier2_compression: int = 16
    salience_hidden: int = 256
    type_priors: Dict[str, float] = None
    tau_threshold: float = 0.8
    
    def __post_init__(self):
        if self.type_priors is None:
            self.type_priors = {
                'NAMED_ENTITY': 1.0,
                'NUMERIC': 1.0,
                'CONTENT_WORD': 0.7,
                'FUNCTION_WORD': 0.1,
                'PUNCTUATION': 0.0,
                'OTHER': 0.5,
            }


class SalienceScorer(nn.Module):
    def __init__(self, hidden_dim: int = 768, salience_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, salience_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(salience_hidden, salience_hidden // 2),
            nn.ReLU(),
            nn.Linear(salience_hidden // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)
    
    def save_pretrained(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_pretrained(cls, path: str, hidden_dim: int = 768, salience_hidden: int = 256) -> 'SalienceScorer':
        instance = cls(hidden_dim=hidden_dim, salience_hidden=salience_hidden)
        instance.load_state_dict(torch.load(path, weights_only=True))
        instance.eval()
        return instance


class TieredKVCache:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.clear()
    
    def clear(self):
        self.k_cache = []
        self.v_cache = []
        self.retention_scores = []
        self.positions = []
        self.total_tokens = 0
    
    def add(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor, positions: torch.Tensor):
        self.k_cache.append(k)
        self.v_cache.append(v)
        self.retention_scores.append(retention)
        self.positions.append(positions)
        self.total_tokens += k.size(2)
    
    def _extract_and_stack(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        if not tensor_list:
            return torch.empty(0)
        
        # Handle different dimensionalities
        if tensor_list[0].dim() == 1:
            # 1D position tensors: [len] -> need to pad to same length
            max_len = max(t.shape[0] for t in tensor_list)
            padded = []
            for t in tensor_list:
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    t = torch.cat([t, torch.zeros(pad_len, device=t.device, dtype=t.dtype)], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0) if padded else None
        elif tensor_list[0].dim() == 3:
            # 3D KV tensors: [heads, len, head_dim]
            max_len = max(t.shape[1] for t in tensor_list)
            padded = []
            for t in tensor_list:
                pad_len = max_len - t.shape[1]
                if pad_len > 0:
                    pad_shape = (t.shape[0], pad_len, t.shape[2])
                    t = torch.cat([t, torch.zeros(*pad_shape, device=t.device, dtype=t.dtype)], dim=1)
                padded.append(t)
            return torch.stack(padded, dim=0) if padded else None
        else:
            raise ValueError(
                f"Unsupported tensor dimensionality: {tensor_list[0].dim()}. "
                f"Expected 1D (position tensors) or 3D (KV tensors)."
            )
    
    def get_compressed_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.k_cache:
            return None, None, None
        
        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        retention_all = torch.cat(self.retention_scores, dim=1)
        positions_all = torch.cat(self.positions, dim=1)
        
        batch_size, num_heads, total_len, head_dim = k_all.shape
        device = k_all.device
        tau = self.config.tau_threshold
        
        protected_mask = retention_all > tau
        unprotected_mask = ~protected_mask
        
        k_tiers = []
        v_tiers = []
        pos_tiers = []
        
        if protected_mask.any():
            k_prot_list, v_prot_list, pos_prot_list = [], [], []
            for b in range(batch_size):
                mask = protected_mask[b]
                if mask.any():
                    k_prot_list.append(k_all[b, :, mask, :])
                    v_prot_list.append(v_all[b, :, mask, :])
                    pos_prot_list.append(positions_all[b, mask])
            
            if k_prot_list:
                k_tiers.append(self._extract_and_stack(k_prot_list))
                v_tiers.append(self._extract_and_stack(v_prot_list))
                pos_tiers.append(self._extract_and_stack(pos_prot_list))
        
        recent_mask = unprotected_mask.clone()
        for b in range(batch_size):
            recent_mask[b] = unprotected_mask[b] & (torch.arange(total_len, device=device) < self.config.tier0_size)
        
        if recent_mask.any():
            k_rec_list, v_rec_list, pos_rec_list = [], [], []
            for b in range(batch_size):
                mask = recent_mask[b]
                if mask.any():
                    k_rec_list.append(k_all[b, :, mask, :])
                    v_rec_list.append(v_all[b, :, mask, :])
                    pos_rec_list.append(positions_all[b, mask])
            
            if k_rec_list:
                k_tiers.append(self._extract_and_stack(k_rec_list))
                v_tiers.append(self._extract_and_stack(v_rec_list))
                pos_tiers.append(self._extract_and_stack(pos_rec_list))
        
        middle_mask = unprotected_mask.clone()
        for b in range(batch_size):
            idx = torch.arange(total_len, device=device)
            middle_mask[b] = unprotected_mask[b] & (idx >= self.config.tier0_size) & (idx < self.config.tier1_size)
        
        if middle_mask.any():
            k_mid_list, v_mid_list, pos_mid_list = [], [], []
            for b in range(batch_size):
                mask = middle_mask[b]
                if mask.any():
                    k_mid_list.append(k_all[b, :, mask, :])
                    v_mid_list.append(v_all[b, :, mask, :])
                    pos_mid_list.append(positions_all[b, mask])
            
            if k_mid_list:
                k_mid = self._extract_and_stack(k_mid_list)
                v_mid = self._extract_and_stack(v_mid_list)
                pos_mid = self._extract_and_stack(pos_mid_list)
                ret_mid = retention_all[middle_mask].view(batch_size, -1)[:, :k_mid.shape[2]]
                
                k_comp, v_comp, pos_comp = self._compress(
                    k_mid, v_mid, ret_mid, pos_mid, self.config.tier1_compression
                )
                k_tiers.append(k_comp)
                v_tiers.append(v_comp)
                pos_tiers.append(pos_comp)
        
        old_mask = unprotected_mask.clone()
        for b in range(batch_size):
            old_mask[b] = unprotected_mask[b] & (torch.arange(total_len, device=device) >= self.config.tier1_size)
        
        if old_mask.any():
            k_old_list, v_old_list, pos_old_list = [], [], []
            for b in range(batch_size):
                mask = old_mask[b]
                if mask.any():
                    k_old_list.append(k_all[b, :, mask, :])
                    v_old_list.append(v_all[b, :, mask, :])
                    pos_old_list.append(positions_all[b, mask])
            
            if k_old_list:
                k_old = self._extract_and_stack(k_old_list)
                v_old = self._extract_and_stack(v_old_list)
                pos_old = self._extract_and_stack(pos_old_list)
                ret_old = retention_all[old_mask].view(batch_size, -1)[:, :k_old.shape[2]]
                
                k_comp, v_comp, pos_comp = self._compress(
                    k_old, v_old, ret_old, pos_old, self.config.tier2_compression
                )
                k_tiers.append(k_comp)
                v_tiers.append(v_comp)
                pos_tiers.append(pos_comp)
        
        if k_tiers:
            k_cat = torch.cat(k_tiers, dim=2)
            v_cat = torch.cat(v_tiers, dim=2)
            pos_cat = torch.cat(pos_tiers, dim=1)
            sort_idx = torch.argsort(pos_cat, dim=1)
            k_sorted = torch.gather(k_cat, 2, sort_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim))
            v_sorted = torch.gather(v_cat, 2, sort_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim))
            pos_sorted = torch.gather(pos_cat, 1, sort_idx)
            return k_sorted, v_sorted, pos_sorted
        else:
            return (torch.empty(batch_size, num_heads, 0, head_dim, device=device),
                    torch.empty(batch_size, num_heads, 0, head_dim, device=device),
                    torch.empty(batch_size, 0, device=device, dtype=torch.long))
    
    def _compress(self, k: torch.Tensor, v: torch.Tensor, retention: torch.Tensor,
                  positions: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, head_dim = k.shape
        device = k.device
        
        if seq_len == 0:
            return k, v, positions
        
        compressed_len = (seq_len + ratio - 1) // ratio
        k_out, v_out, pos_out = [], [], []
        
        for b in range(batch_size):
            k_batch, v_batch, pos_batch = [], [], []
            for i in range(compressed_len):
                start = i * ratio
                end = min((i + 1) * ratio, seq_len)
                
                k_chunk = k[b, :, start:end, :]
                v_chunk = v[b, :, start:end, :]
                ret_chunk = retention[b, start:end]
                pos_chunk = positions[b, start:end]
                
                weights = F.softmax(ret_chunk, dim=0).unsqueeze(0).unsqueeze(-1)
                k_pooled = (k_chunk * weights).sum(dim=1)
                v_pooled = (v_chunk * weights).sum(dim=1)
                pos_pooled = (pos_chunk.float() * weights.squeeze()).sum().long()
                
                k_batch.append(k_pooled)
                v_batch.append(v_pooled)
                pos_batch.append(pos_pooled)
            
            k_out.append(torch.stack(k_batch, dim=1))
            v_out.append(torch.stack(v_batch, dim=1))
            pos_out.append(torch.stack(pos_batch))
        
        return torch.stack(k_out, dim=0), torch.stack(v_out, dim=0), torch.stack(pos_out, dim=0)
    
    def get_stats(self) -> Dict:
        if not self.k_cache:
            return {'total_tokens': 0, 'compressed_tokens': 0, 'compression_ratio': 1.0}
        
        k_comp, _, _ = self.get_compressed_cache()
        if k_comp is None:
            return {'total_tokens': self.total_tokens, 'compressed_tokens': 0, 'compression_ratio': float(self.total_tokens)}
        
        compressed_len = k_comp.size(2)
        
        return {
            'total_tokens': self.total_tokens,
            'compressed_tokens': compressed_len,
            'compression_ratio': self.total_tokens / max(compressed_len, 1)
        }
