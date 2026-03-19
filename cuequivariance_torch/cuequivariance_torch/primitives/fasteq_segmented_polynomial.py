# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by mlx in 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Dict, List, Optional, OrderedDict, Tuple, Any

import torch
import torch.nn as nn
from cuequivariance_torch.primitives.segmented_polynomial_fused_tp import (
    SegmentedPolynomialFusedTP,
)
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SegmentedPolynomialIndexedLinear,
)
from cuequivariance_torch.primitives.segmented_polynomial_naive import (
    SegmentedPolynomialNaive,
)
from cuequivariance_torch.primitives.segmented_polynomial_uniform_1d import (
    SegmentedPolynomialFromUniform1dJit,
)

import cuequivariance as cue

try:
    import cuequivariance_ops_torch  # noqa: F401

    HAS_CUE_OPS = True
except ImportError:
    HAS_CUE_OPS = False

import time
#from mace.tools.scatter import scatter_sum

from fasteq.ops.equi_linear import fast_equi_linear
from fasteq.ops.stc import fast_stc
from fasteq.ops.cwtp import fast_cwtp
from fasteq.ops.mptp import fast_mptp
from fasteq.ops.fctp import fast_fctp
from fasteq.ops.uniform1d_fused import fast_uniform1d_fused
from fasteq.ops.uniform1d_jit import fast_uniform1d_jit

import math
from torch.nn.utils.rnn import pad_sequence

def flatten_stp(d: cue.SegmentedTensorProduct) -> cue.SegmentedTensorProduct:
    

    d = d.move_operand(0, -2)
    d = d.flatten_coefficient_modes(force=True)
    d = d.flatten_modes(
        [
            m
            for m in d.subscripts.modes()
            if not all(m in ss for ss in d.subscripts.operands)
        ]
    )
    d = d.consolidate_modes()
    if d.subscripts.modes() == []:
        d = d.append_modes_to_all_operands("u", dict(u=1))
    '''
    for oid in range(0, d.num_operands - 2):
        print(f"oid:{oid}, len d.operands[oid].num_segments:{d.operands[oid].num_segments}")
    '''

    # ops.SymmetricTensorContraction will "symmetrize" for the derivatives so we can sort for the forward pass
    d = d.sort_indices_for_identical_operands(range(0, d.num_operands - 2))

    if len(d.subscripts.modes()) != 1:
        raise NotImplementedError("Different modes are not supported.")

    m = d.subscripts.modes()[0]

    if not all(ss == m for ss in d.subscripts.operands):
        raise NotImplementedError("Different subscripts are not supported.")

    d = d.split_mode(m, math.gcd(*d.get_dims(m)))

    return d

@torch.no_grad()
def infer_cwtp_meta(
    descriptor,
    math_dtype,
    device,
    max_k_dim_for_kernel: int = 8, # TODO: make it flexible
) -> Dict[str, Any]:
    """
    生成 ChannelWise TP 的所有 meta 信息：
      - 分段信息：uv / iu / jv / kv offsets + slices
      - dense c_tensors + (i/j/k_dims, c_offsets, c_all)
      - sparse CG 信息：
        * cg_i_all, cg_j_all, cg_k_all, cg_val_all
        * nnz_per_path, nnz_offsets
      - 按 k 分组的 sparse meta：
        * nnz_k_offsets: [P, MAX_K_DIM]
        * nnz_k_counts:  [P, MAX_K_DIM]
    """

    # -------------------- 1) paths & dense c_tensors --------------------
    path_indices: List[Tuple[int, int, int, int]] = []
    c_tensors: List[torch.Tensor] = []

    for path_idx, path in enumerate(descriptor.paths):
        path_indices.append(tuple(path.indices))

        c_tensor = torch.tensor(path.coefficients, dtype=math_dtype, device=device).contiguous()
        c_tensors.append(c_tensor)

    # U/V
    u = list(descriptor.get_dims("u"))[0]
    v = list(descriptor.get_dims("v"))[0]

    UV_TOTAL = int(descriptor.operands[0].size)
    IU_TOTAL = int(descriptor.operands[1].size)
    JV_TOTAL = int(descriptor.operands[2].size)

    '''
    for ops in descriptor.operands:
        print(f"==== ops: {ops} ====")
        for seg in ops.segments:
            print(f"seg: {seg}")
    '''


    # segment counts
    uv_seg_count = max(p[0] for p in path_indices) + 1
    iu_seg_count = max(p[1] for p in path_indices) + 1
    jv_seg_count = max(p[2] for p in path_indices) + 1
    kv_seg_count = max(p[3] for p in path_indices) + 1

    # -------------------- 2) i/j/k dims + c_offsets + c_all --------------------
    i_dims, j_dims, k_dims = [], [], []
    c_offsets = [0]
    c_flat_list = []
    running = 0

    for c in c_tensors:
        i_dim, j_dim, k_dim = map(int, c.shape)
        i_dims.append(i_dim)
        j_dims.append(j_dim)
        k_dims.append(k_dim)

        c_flat = c.reshape(-1).contiguous()          # original layout (i,j,k) flatten => ((i*J + j)*K + k)
        c_flat_list.append(c_flat)

        running += i_dim * j_dim * k_dim
        c_offsets.append(running)

    c_all = (
        torch.cat(c_flat_list, dim=0)
        if len(c_flat_list) > 0
        else torch.empty(0, dtype=math_dtype, device=device)
    )

    max_k_dim = max(k_dims) if len(k_dims) > 0 else 0
    assert max_k_dim <= max_k_dim_for_kernel, \
        f"max k_dim {max_k_dim} > kernel MAX_K_DIM {max_k_dim_for_kernel}"

    # -------------------- 3) uv slices + uv_seg_offsets --------------------
    uv_slices = []
    uv_seg_offsets = []
    start = 0
    uv_stride = int(u * v)
    for _ in range(uv_seg_count):
        uv_seg_offsets.append(start)
        end = start + uv_stride
        uv_slices.append(slice(start, end))
        start = end
    assert start == UV_TOTAL, f"UV total {start} != {UV_TOTAL}"

    # -------------------- 4) iu slices + iu_seg_offsets --------------------
    iu_slices = []
    iu_seg_offsets = []
    start = 0
    for s in range(iu_seg_count):
        iu_seg_offsets.append(start)
        idx = next(idx for idx, p in enumerate(path_indices) if p[1] == s)
        i_dim = i_dims[idx]
        length = int(i_dim * u)
        end = start + length
        iu_slices.append(slice(start, end))
        start = end
    assert start == IU_TOTAL, f"IU total {start} != {IU_TOTAL}"

    # -------------------- 5) jv slices + jv_seg_offsets --------------------
    jv_slices = []
    jv_seg_offsets = []
    start = 0
    for s in range(jv_seg_count):
        jv_seg_offsets.append(start)
        idx = next(idx for idx, p in enumerate(path_indices) if p[2] == s)
        j_dim = j_dims[idx]
        length = int(j_dim * v)
        end = start + length
        jv_slices.append(slice(start, end))
        start = end
    assert start == JV_TOTAL, f"JV total {start} != {JV_TOTAL}"

    # -------------------- 6) K offsets: kv_k_offsets --------------------
    kv_k_offsets = []
    start = 0
    for s in range(kv_seg_count):
        kv_k_offsets.append(start)
        idx = next(idx for idx, p in enumerate(path_indices) if p[3] == s)
        k_dim = k_dims[idx]
        start += int(k_dim)
    K_TOTAL = int(start)

    # -------------------- 7) sparse CG meta + per-k grouping --------------------
    cg_i_list, cg_j_list, cg_k_list, cg_val_list = [], [], [], []
    nnz_per_path = []
    nnz_offsets = [] 

    nnz_k_offsets_list = []  # [P, MAX_K_DIM]
    nnz_k_counts_list  = []  # [P, MAX_K_DIM]

    nnz_running = 0
    for path_id, c in enumerate(c_tensors):
        i_dim, j_dim, k_dim = map(int, c.shape)

        nz_idx = torch.nonzero(c != 0, as_tuple=False)  # [nnz,3] (i,j,k)
        nnz = int(nz_idx.size(0))

        #print(f"cwtp nnz path:{path_id}, nnz idx:{nz_idx}")

        nnz_per_path.append(nnz)
        nnz_offsets.append(nnz_running)
        nnz_running += nnz

        local_k_offsets = torch.zeros(max_k_dim_for_kernel, dtype=torch.int32, device=device)
        local_k_counts  = torch.zeros(max_k_dim_for_kernel, dtype=torch.int32, device=device)

        if nnz > 0:
            sort_idx = torch.argsort(nz_idx[:, 2])  # sort by k
            nz_sorted = nz_idx[sort_idx]
            i_idx = nz_sorted[:, 0]
            j_idx = nz_sorted[:, 1]
            k_idx = nz_sorted[:, 2]


            vals = c[i_idx, j_idx, k_idx]

            cg_i_list.append(i_idx.to(torch.uint8))
            cg_j_list.append(j_idx.to(torch.uint8))
            cg_k_list.append(k_idx.to(torch.uint8))
            cg_val_list.append(vals)

            prev_k = int(k_idx[0].item())
            local_k_offsets[prev_k] = 0

            for t in range(1, nnz):
                curr_k = int(k_idx[t].item())
                if curr_k != prev_k:
                    local_k_counts[prev_k] = t - int(local_k_offsets[prev_k].item())
                    local_k_offsets[curr_k] = t
                    prev_k = curr_k

            local_k_counts[prev_k] = nnz - int(local_k_offsets[prev_k].item())

        nnz_k_offsets_list.append(local_k_offsets)
        nnz_k_counts_list.append(local_k_counts)

    if len(cg_i_list) > 0:
        cg_i_all = torch.cat(cg_i_list, dim=0).contiguous()
        cg_j_all = torch.cat(cg_j_list, dim=0).contiguous()
        cg_k_all = torch.cat(cg_k_list, dim=0).contiguous()
        cg_val_all = torch.cat(cg_val_list, dim=0).contiguous()
    else:
        cg_i_all = torch.empty(0, dtype=torch.uint8, device=device)
        cg_j_all = torch.empty(0, dtype=torch.uint8, device=device)
        cg_k_all = torch.empty(0, dtype=torch.uint8, device=device)
        cg_val_all = torch.empty(0, dtype=math_dtype, device=device)

    nnz_per_path_t = torch.tensor(nnz_per_path, dtype=torch.int32, device=device)
    nnz_offsets_t  = torch.tensor(nnz_offsets,  dtype=torch.int32, device=device)

    nnz_k_offsets = torch.stack(nnz_k_offsets_list, dim=0).contiguous()  # [P, MAX_K_DIM]
    nnz_k_counts  = torch.stack(nnz_k_counts_list,  dim=0).contiguous()  # [P, MAX_K_DIM]
    nnz_k_offsets_flat = nnz_k_offsets.reshape(-1).contiguous()
    nnz_k_counts_flat  = nnz_k_counts.reshape(-1).contiguous()

    # -------------------- 8) pack tensors for kernels --------------------
    path_indices_tensor = torch.tensor(path_indices, dtype=torch.int32, device=device).contiguous()
    i_dims_t = torch.tensor(i_dims, dtype=torch.int32, device=device).contiguous()
    j_dims_t = torch.tensor(j_dims, dtype=torch.int32, device=device).contiguous()
    k_dims_t = torch.tensor(k_dims, dtype=torch.int32, device=device).contiguous()
    c_offsets_t = torch.tensor(c_offsets, dtype=torch.int32, device=device).contiguous()  # [P+1]

    meta = {
        # dense
        "c_tensors": c_tensors,
        "path_indices": path_indices,

        # dense packed (original i-j-k flatten)
        "path_indices_tensor": path_indices_tensor,  # [P,4] int32
        "c_all": c_all,                              # [sum(i*j*k)] layout ((i*J+j)*K+k)
        "c_offsets": c_offsets_t,                    # [P+1]

        # slices (reference)
        "uv_slices": uv_slices,
        "iu_slices": iu_slices,
        "jv_slices": jv_slices,

        # offsets
        "uv_seg_offsets": torch.tensor(uv_seg_offsets, dtype=torch.int32, device=device).contiguous(),
        "iu_seg_offsets": torch.tensor(iu_seg_offsets, dtype=torch.int32, device=device).contiguous(),
        "jv_seg_offsets": torch.tensor(jv_seg_offsets, dtype=torch.int32, device=device).contiguous(),
        "kv_k_offsets": torch.tensor(kv_k_offsets, dtype=torch.int32, device=device).contiguous(),

        # dims
        "i_dims": i_dims_t,
        "j_dims": j_dims_t,
        "k_dims": k_dims_t,

        # sizes
        "U": int(u),
        "V": int(v),
        "UV_TOTAL": UV_TOTAL,
        "IU_TOTAL": IU_TOTAL,
        "JV_TOTAL": JV_TOTAL,
        "K_TOTAL": K_TOTAL,

        # sparse CG info
        "cg_i_all": cg_i_all,
        "cg_j_all": cg_j_all,
        "cg_k_all": cg_k_all,
        "cg_val_all": cg_val_all,
        "nnz_per_path": nnz_per_path_t,
        "nnz_offsets": nnz_offsets_t,  # length P (start offset per path)

        # sparse per-k grouping
        "nnz_k_offsets": nnz_k_offsets_flat,  # [P*MAX_K_DIM]
        "nnz_k_counts": nnz_k_counts_flat,    # [P*MAX_K_DIM]
        "MAX_K_DIM": int(max_k_dim_for_kernel),
    }
    return meta

@torch.no_grad()
def build_grouped_paths(path_segment_indices: torch.Tensor,
                        path_coefficients: torch.Tensor,
                        V: int,
                        device=None):
    """
    path_segment_indices: [P,4] int64/int32, columns: i,j,k,v
    path_coefficients:    [P]   float32/float64
    returns:
      i_list, j_list, k_list: [P] int32
      coeff_list:             [P] same dtype as coefficients
      v_offsets:              [V+1] int32, CSR-like offsets into lists
    """
    assert path_segment_indices.ndim == 2 and path_segment_indices.size(1) == 4
    P = path_segment_indices.size(0)
    if device is None:
        device = path_segment_indices.device

    psi = path_segment_indices.to("cpu")
    coeff = path_coefficients.to("cpu")

    v = psi[:, 3].to(torch.int64)
    # sort by v
    order = torch.argsort(v, stable=True)
    psi_s = psi[order]
    coeff_s = coeff[order]

    v_s = psi_s[:, 3].to(torch.int64)
    counts = torch.bincount(v_s, minlength=V)
    v_offsets = torch.zeros(V + 1, dtype=torch.int32)
    v_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    i_list = psi_s[:, 0].to(torch.int32).contiguous().to(device)
    j_list = psi_s[:, 1].to(torch.int32).contiguous().to(device)
    k_list = psi_s[:, 2].to(torch.int32).contiguous().to(device)
    v_list = psi_s[:, 3].to(torch.int32).contiguous().to(device)
    coeff_list = coeff_s.contiguous().to(device)
    v_offsets = v_offsets.contiguous().to(device)

    return i_list, j_list, k_list, v_list, coeff_list, v_offsets

@torch.no_grad()
def build_csr_buckets(cls_idx: torch.Tensor, S: int):
    """
    cls_idx: [B] int32/int64, on CUDA, values in [0..S-1]
    returns:
      b_list: [B] int32 CUDA
      cls_offsets: [S+1] int32 CUDA
    """
    assert cls_idx.is_cuda
    order = torch.argsort(cls_idx.to(torch.int64), stable=True)   # [B]
    b_list = order.to(torch.int32)

    counts = torch.bincount(cls_idx.to(torch.int64), minlength=S) # [S] on CUDA
    cls_offsets = torch.empty(S + 1, device=cls_idx.device, dtype=torch.int32)
    cls_offsets[0] = 0
    cls_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return b_list, cls_offsets

@torch.no_grad()
def build_k_sliced_ell_packed(
    num_paths: int,
    MAX_K_DIM: int,
    k_dims: torch.Tensor,          # [num_paths] int32
    nnz_offsets: torch.Tensor,     # [num_paths] int32
    nnz_k_offsets: torch.Tensor,   # [num_paths * MAX_K_DIM] int32
    nnz_k_counts: torch.Tensor,    # [num_paths * MAX_K_DIM] int32
    cg_i_all: torch.Tensor,        # [nnz_total] uint8
    cg_j_all: torch.Tensor,        # [nnz_total] uint8
    cg_val_all: torch.Tensor,      # [nnz_total] float/double
    sort_within_k: bool = True,    # 可选：每个k内按(i,j)排序
):
    """
    输出：
      ell_E    : [num_paths] int32，E[p]=max nnz per k
      ell_base : [num_paths] int32，ell_ij/ell_val 的起点（以 element 为单位）
      ell_ij   : [ell_total] uint16，按 (k_local * E + e) 排布
      ell_val  : [ell_total] scalar，同上
    """


    assert k_dims.dtype == torch.int32
    assert nnz_offsets.dtype == torch.int32
    assert nnz_k_offsets.dtype == torch.int32
    assert nnz_k_counts.dtype == torch.int32
    assert cg_i_all.dtype == torch.uint8 and cg_j_all.dtype == torch.uint8
    assert cg_val_all.dtype in (torch.float16, torch.float32, torch.float64)

    device = cg_val_all.device
    k_dims_cpu        = k_dims.cpu()
    nnz_offsets_cpu   = nnz_offsets.cpu()
    nnz_k_offsets_cpu = nnz_k_offsets.cpu()
    nnz_k_counts_cpu  = nnz_k_counts.cpu()
    cg_i_cpu          = cg_i_all.cpu()
    cg_j_cpu          = cg_j_all.cpu()
    cg_val_cpu        = cg_val_all.cpu()

    # 1) 计算 E[p]
    ell_E_cpu = torch.empty((num_paths,), dtype=torch.int32)
    for p in range(num_paths):
        k_dim = int(k_dims_cpu[p].item())
        if k_dim <= 0:
            ell_E_cpu[p] = 0
            continue
        counts = nnz_k_counts_cpu[p * MAX_K_DIM : p * MAX_K_DIM + k_dim]
        ell_E_cpu[p] = int(counts.max().item()) if counts.numel() > 0 else 0

    # 2) 计算 base 偏移（prefix sum）
    ell_base_cpu = torch.empty((num_paths,), dtype=torch.int32)
    total = 0
    for p in range(num_paths):
        ell_base_cpu[p] = total
        k_dim = int(k_dims_cpu[p].item())
        E = int(ell_E_cpu[p].item())
        total += k_dim * E

    # 3) 分配 ELL 数组（pad: val=0，ij=0）
    ell_ij_cpu  = torch.zeros((total,), dtype=torch.uint16)
    ell_val_cpu = torch.zeros((total,), dtype=cg_val_cpu.dtype)

    # 4) 填充
    for p in range(num_paths):
        k_dim = int(k_dims_cpu[p].item())
        if k_dim <= 0:
            continue
        E = int(ell_E_cpu[p].item())
        if E <= 0:
            continue

        base = int(ell_base_cpu[p].item())
        nnz_off = int(nnz_offsets_cpu[p].item())

        for k_local in range(k_dim):
            meta_idx = p * MAX_K_DIM + k_local
            local_off = int(nnz_k_offsets_cpu[meta_idx].item())
            local_cnt = int(nnz_k_counts_cpu[meta_idx].item())
            # 该k的输出行起点
            row = base + k_local * E

            if local_cnt <= 0:
                continue

            # 拿到该k的 nnz 索引范围：idx = nnz_off + (local_off + tt)
            idxs = torch.arange(local_off, local_off + local_cnt, dtype=torch.int32)
            gidx = (nnz_off + idxs).to(torch.int64)  # 作为索引用 int64

            ii = cg_i_cpu[gidx].to(torch.int32)
            jj = cg_j_cpu[gidx].to(torch.int32)
            vv = cg_val_cpu[gidx]

            if sort_within_k:
                # sort key: i major then j
                key = ii * 256 + jj
                order = torch.argsort(key)
                ii = ii[order]; jj = jj[order]; vv = vv[order]

            # 写入前 local_cnt 个，其余 pad=0
            w = min(local_cnt, E)
            # pack: ij = i | (j<<8)
            ij = (ii[:w] & 0xFF) | ((jj[:w] & 0xFF) << 8)
            ell_ij_cpu[row : row + w]  = ij.to(torch.uint16)
            ell_val_cpu[row : row + w] = vv[:w]

            # pad 部分已是0：val=0 => acc +=0，可无分支

    # 5) 搬回 device（通常是 GPU）
    ell_E    = ell_E_cpu.to(device=device)
    ell_base = ell_base_cpu.to(device=device)
    ell_ij   = ell_ij_cpu.to(device=device)
    ell_val  = ell_val_cpu.to(device=device)
    ell_meta = {
        "ell_E": ell_E,
        "ell_base": ell_base,
        "ell_ij": ell_ij,
        "ell_val": ell_val,
    }
    return ell_meta

@torch.no_grad()
def build_packed_meta(path_indices: torch.Tensor,
                      k_dims: torch.Tensor,
                      ell_E: torch.Tensor,
                      ell_base: torch.Tensor,
                      iu_seg_offsets: torch.Tensor,
                      jv_seg_offsets: torch.Tensor,
                      kv_k_offsets: torch.Tensor,
                      U: int):
    """
    path_indices: [P,4] int32 (uv_idx, iu_idx, jv_idx, kv_idx)
    k_dims:      [P]   int32
    ell_E:       [P]   int32
    ell_base:    [P]   int32
    offsets: small fixed int32 tensors on GPU
    returns:
      meta1: [P,4] int32  (uv_base, iu_base, jv_base, k_base)
      meta2: [P,4] int32  (k_dim, E, ell_base, pad)
    """
    assert path_indices.dtype == torch.int32 and path_indices.is_cuda
    P = path_indices.shape[0]

    uv_idx = path_indices[:, 0]
    iu_idx = path_indices[:, 1]
    jv_idx = path_indices[:, 2]
    kv_idx = path_indices[:, 3]

    uv_base = uv_idx * U                         # V==1, x_uv is [K][U]
    iu_base = iu_seg_offsets[iu_idx]             # gather from tiny fixed offsets
    jv_base = jv_seg_offsets[jv_idx]
    k_base  = kv_k_offsets[kv_idx]

    meta1 = torch.stack([uv_base, iu_base, jv_base, k_base], dim=1).contiguous()  # [P,4] int32
    pad   = torch.zeros((P,), device=meta1.device, dtype=torch.int32)
    meta2 = torch.stack([k_dims, ell_E, ell_base, pad], dim=1).contiguous()       # [P,4] int32

    return meta1, meta2

@torch.no_grad()
def make_p_for_k(K_per_path, device):
    K_total = sum(K_per_path)
    p_for_k = torch.empty((K_total,), device=device, dtype=torch.int32)
    off = 0
    for p, kp in enumerate(K_per_path):
        p_for_k[off:off+kp] = p
        off += kp
    return p_for_k


@torch.no_grad()
def make_cg_single_mapping(K_total, I_total, device, path_num, math_dtype):
    """
    Build i_for_k[K], val_for_k[K], each k has at most one nnz (or empty).
      diag    : i_for_k[k]=k if k<I_total
      single0 : only k=0 uses i=0
    """
    i_for_k = torch.full((K_total,), -1, device=device, dtype=torch.int32)
    val_for_k = torch.zeros((K_total,), device=device, dtype=math_dtype)

    # "diag"
    if path_num == 4:
        kk = torch.arange(K_total, device=device, dtype=torch.int32)
        mask = kk < I_total
        i_for_k[mask] = kk[mask]
        val_for_k[mask] = 1.0
    elif path_num == 1:
        if K_total > 0:
            i_for_k[0] = 0
            val_for_k[0] = 1.0
    else:
        raise ValueError(f"Unknown path_num: {path_num}")

    return i_for_k, val_for_k

@torch.no_grad()
def infer_fctp_meta(descriptor, math_dtype, device):
    # per-path tensors
    cg_indices = []
    cg_values  = []
    c_tensors  = []
    dim_list   = []

    #print(f"i dims:", sum(descriptor.get_dims("i")))
    #print(f"j dims:", sum(descriptor.get_dims("j")))
    #print(f"k dims:", sum(descriptor.get_dims("k")))

    # 1) build per-path (idx, val, coeffs)
    for i, path in enumerate(descriptor.paths):
        if getattr(path, "coefficients", None) is None or path.coefficients.ndim < 3:
            raise ValueError("FCTP only supports paths with explicit 3D coefficient tensors.")

        coeffs = torch.from_numpy(path.coefficients).to(device=device, dtype=math_dtype)

        # idx: [nnz, 3] (i,j,k) ; vals: [nnz]
        idx = coeffs.nonzero(as_tuple=False).to(device=device, dtype=torch.int32)
        vals = coeffs[idx[:, 0], idx[:, 1], idx[:, 2]].to(device=device, dtype=math_dtype)

        dim_list.append(int(vals.numel()))

        cg_indices.append(idx)
        cg_values.append(vals)
        c_tensors.append(coeffs)

    # 2) global dimensions
    dimensions_dict = descriptor.get_dimensions_dict()
    U = sum(dimensions_dict["u"])
    V = sum(dimensions_dict["v"])
    W = sum(dimensions_dict["w"])

    # 3) K_per_path / offsets / totals
    P = len(cg_indices)
    assert P == len(dim_list)

    K_per_path = torch.tensor(dim_list, device=device, dtype=torch.int32)
    path_offset = torch.empty(P, device=device, dtype=torch.int32)
    path_offset[0] = 0
    if P > 1:
        path_offset[1:] = torch.cumsum(K_per_path[:-1], dim=0)
    
    #K_total = int(K_per_path.sum().item())
    I_total = sum(descriptor.get_dims("i"))
    K_total = sum(descriptor.get_dims("k"))


    # 4) pack nnz info
    nnz_list = [int(ci.shape[0]) for ci in cg_indices]
    nnz_max = max(nnz_list) if nnz_list else 0
    nnz_per_path = torch.tensor(nnz_list, device=device, dtype=torch.int32)

    # 5) pack cg_*_all
    cg_i_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_j_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_k_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_val_all = torch.zeros((P, nnz_max), device=device, dtype=math_dtype)

    for p in range(P):
        ci_local = cg_indices[p]   # [nnz_p, 3]
        cv       = cg_values[p]    # [nnz_p]
        nnz_p    = nnz_list[p]
        offset_p = int(path_offset[p].item())

        i_local = ci_local[:, 0]
        j_local = ci_local[:, 1]
        k_local = ci_local[:, 2]

        # --- local -> global ---
        i_global = i_local + offset_p
        j_global = j_local              # TODO: only support j_local = j_global = 0
        k_global = k_local + offset_p

        cg_i_all[p, :nnz_p]   = i_global
        cg_j_all[p, :nnz_p]   = j_global
        cg_k_all[p, :nnz_p]   = k_global
        cg_val_all[p, :nnz_p] = cv
    
    i_for_k, val_for_k = make_cg_single_mapping(K_total, I_total, device, P, math_dtype)
    p_for_k = make_p_for_k(K_per_path, device)

    return {
        "cg_indices": cg_indices,
        "cg_values": cg_values,
        "c_tensors": c_tensors,

        "U": U, "V": V, "W": W,

        "P": P,
        "K_per_path": K_per_path,
        "path_offset": path_offset,
        "I_total": I_total,
        "K_total": K_total,

        "nnz_list": nnz_list,
        "nnz_max": nnz_max,
        "nnz_per_path": nnz_per_path,

        "cg_i_all": cg_i_all,
        "cg_j_all": cg_j_all,
        "cg_k_all": cg_k_all,
        "cg_val_all": cg_val_all,
        "i_for_k": i_for_k,
        "val_for_k": val_for_k,
        "p_for_k": p_for_k
    }

class FastEqSegmentedPolynomial(nn.Module):
    """PyTorch module that computes a segmented polynomial.

    Args:
        polynomial: The segmented polynomial to compute, an instance of
            `cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`.
        method: Specifies the implementation method to use. Options are:

            - ``"naive"``: Uses a naive PyTorch implementation. It always works but is not optimized.
            - ``"uniform_1d"``: Uses a CUDA implementation for polynomials with a single uniform mode.
            - ``"fused_tp"``: Uses a CUDA implementation for polynomials with 3- and 4-operand contractions.
            - ``"indexed_linear"``: Uses a CUDA implementation for linear layers with indexed weights.

        math_dtype: Optional data type for computational operations.
            If specified, internal buffers will be of this dtype,
            and operands will be converted to this type for all computations.

            Values can be specified as a string corresponding to a torch.dtype,
            or as a torch.dtype.
            For some methods, special values can be used:

            - For method ``"naive"``: Any torch.dtype or corresponding string.
            - For method ``"uniform_1d"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"fused_tp"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"indexed_linear"``: this is not supported and will be ignored.

            .. note::
               This will not be affected by changes to the module dtype,
               and not all methods support all dtypes.

            If ``math_dtype`` is not specified:

            - For method ``"naive"``, the dtype of the input tensors will be used.
            - For method ``"uniform_1d"``, the dtype of the input tensors will be used if allowed
              (FP32 or FP64), otherwise float32 will be used.
            - For method ``"fused_tp"``, the default dtype (FP32) will be used.
            - For method ``"indexed_linear"``, the dtype of the input tensors will be used.

        output_dtype_map: Optional list that, for each output buffer, specifies
            the index of the input buffer from which it inherits its data type.
            -1 means the math_dtype is used.
            Default is 0 if there are input tensors, otherwise -1.
        name: Optional name for the operation. Defaults to "segmented_polynomial".

    Examples:
        Basic usage with spherical harmonics:

        >>> import torch
        >>> import cuequivariance as cue
        >>> from cuequivariance_torch import SegmentedPolynomial
        >>>
        >>> # Create spherical harmonics polynomial
        >>> poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial
        >>> sp = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Compute spherical harmonics for unit vector along y-axis
        >>> x = torch.tensor([[0.0, 1.0, 0.0]])
        >>> result = sp([x])
        >>> print(result[0].shape)
        torch.Size([1, 9])

        Example with a linear layer:

        >>> # Create a linear transformation
        >>> input_irreps = cue.Irreps(cue.O3, "5x0e + 3x1o")
        >>> output_irreps = cue.Irreps(cue.O3, "4x0e + 2x1o")
        >>> poly = cue.descriptors.linear(input_irreps, output_irreps).polynomial
        >>>
        >>> # Create the module
        >>> linear = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Create random weights and input
        >>> weights = torch.randn(1, poly.inputs[0].size)
        >>> x = torch.randn(10, poly.inputs[1].size)
        >>>
        >>> # Forward pass
        >>> result = linear([weights, x])
        >>> print(result[0].shape)
        torch.Size([10, 10])

        Example with indexed operations:

        >>> # Create indexed weights for different elements
        >>> weights = torch.randn(3, poly.inputs[0].size)  # 3 different weight sets
        >>> x = torch.randn(5, poly.inputs[1].size)        # 5 input vectors
        >>>
        >>> # Index tensor specifying which weights to use for each input
        >>> weight_indices = torch.tensor([0, 1, 0, 2, 1])  # Use weights 0,1,0,2,1
        >>>
        >>> result = linear([weights, x],
        ...                input_indices={0: weight_indices})
        >>> print(result[0].shape)
        torch.Size([5, 10])
    """

    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        method: str = "",
        math_dtype: str | torch.dtype = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
        op_name: str = "",
        use_fasteq: Optional[bool] = None,
        u1d_compatible: bool = False,
    ):
        super().__init__()

        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.method = method
        self.repr = polynomial.__repr__()
        self.op_name = op_name
        self.descriptor = polynomial.operations[0][1]
        self.use_fasteq = use_fasteq
        self.polynomial = polynomial
        self.u1d_compatible = u1d_compatible
        
        if method == "":
            warnings.warn(
                "Hello! It looks like you're using code that was written for an older version of this library.\n"
                "Starting in v0.6.0, the `method` argument is suggested when using `SegmentedPolynomial()`.\n"
                "This change helps ensure you get optimal performance by explicitly choosing the computation method.\n"
                "For the moment, we will default to the 'uniform_1d' method.\n\n"
                "To remove this warning, add a `method` parameter to your function call. Here are the available options:\n"
                "• 'naive' - Works everywhere but not optimized (good for testing)\n"
                "• 'uniform_1d' - Fast CUDA implementation for single uniform mode polynomials\n"
                "• 'fused_tp' - A more general CUDA implementation, supporting many 3 and 4 operands contractions.\n"
                "• 'indexed_linear' - A CUDA implementation for linear layers with indexed weights.\n"
            )
            method = "naive"

        if not isinstance(polynomial, cue.SegmentedPolynomial):
            raise ValueError(
                f"The polynomial is not a cue.SegmentedPolynomial, but a {type(polynomial)}",
                "Did you forget to call `.polynomial` on the descriptor?",
            )

        if method != "naive" and not HAS_CUE_OPS:
            method = "naive"
            warnings.warn(
                "cuequivariance_ops_torch is not available. Falling back to naive implementation."
            )

        if method == "uniform_1d":
            self.m = SegmentedPolynomialFromUniform1dJit(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "naive":
            self.m = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "fused_tp":
            self.m = SegmentedPolynomialFusedTP(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
        elif method == "indexed_linear":
            self.m = SegmentedPolynomialIndexedLinear(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        else:
            raise ValueError(f"Invalid method: {method}")

        print(f"Init op:{op_name}, u1d_compatible:{u1d_compatible}, use_fasteq:{use_fasteq}")

        if use_fasteq and (op_name == "stc"):
            ds_ = [flatten_stp(d) for _, d in polynomial.operations]
            d_max = max(ds_, key=lambda d: d.num_operands)
            self.num_out_segments = d_max.operands[-1].num_segments
            self.u = d_max.operands[0].size // d_max.operands[0].num_segments

            path_segment_indices = sum((d.indices.tolist() for  d in ds_), [])
            path_coefficients = sum((d.stacked_coefficients.tolist() for d in ds_), [])

            print(f"op_name:{op_name}, desc:{self.descriptor}")

            device = "cuda" # TODO: make it general
            self.coeffs_tensor = torch.as_tensor(path_coefficients, dtype=math_dtype).to(device)
            self.path_lens_tensor = torch.as_tensor([len(p) for p in path_segment_indices], dtype=torch.int32).to(device)
            self.paths_tensor = pad_sequence(
                [torch.as_tensor(p, dtype=torch.int32) for p in path_segment_indices],
                batch_first=True, padding_value=0
            ).to(device)
        
        elif use_fasteq and op_name == "cwtp" and not u1d_compatible:
            self.meta = infer_cwtp_meta(self.descriptor, math_dtype=math_dtype, device="cuda") # device hardcoded for now
            cg_i_groupk = self.meta["cg_i_all"]
            cg_j_groupk  = self.meta["cg_j_all"]
            cg_val_groupk  = self.meta["cg_val_all"]
            nnz_per_path = self.meta["nnz_per_path"]
            nnz_offsets_groupk = self.meta["nnz_offsets"]
            nnz_k_offsets_groupk = self.meta["nnz_k_offsets"]
            nnz_k_counts_groupk = self.meta["nnz_k_counts"]
            k_dims = self.meta["k_dims"]

            num_paths = nnz_per_path.shape[0]
            self.ell_meta = build_k_sliced_ell_packed(
                num_paths=num_paths,
                MAX_K_DIM=8,
                k_dims=k_dims,                  # int32
                nnz_offsets=nnz_offsets_groupk,         # int32
                nnz_k_offsets=nnz_k_offsets_groupk,     # int32
                nnz_k_counts=nnz_k_counts_groupk,       # int32
                cg_i_all=cg_i_groupk,               # uint8
                cg_j_all=cg_j_groupk,               # uint8
                cg_val_all=cg_val_groupk,           # float/double
                sort_within_k=True,
            )

            uv_seg_offsets = self.meta["uv_seg_offsets"]
            iu_seg_offsets = self.meta["iu_seg_offsets"]
            jv_seg_offsets = self.meta["jv_seg_offsets"]
            kv_k_offsets = self.meta["kv_k_offsets"]
            path_indices = self.meta["path_indices_tensor"]
            U = self.meta["U"]

            meta1, meta2 = build_packed_meta(path_indices, k_dims, self.ell_meta["ell_E"], self.ell_meta["ell_base"], iu_seg_offsets, jv_seg_offsets, kv_k_offsets, U)

            self.ell_meta["meta1"] = meta1
            self.ell_meta["meta2"] = meta2

        elif use_fasteq and ((op_name == "cwtp" and u1d_compatible) or (op_name == "uniform1d")):

            self.descriptor = self.m.polynomial.operations[0][1]
            print(f"cwtp u1d compatible path, descriptor:{self.descriptor}")
            print(f"polynomial.operations:{self.m.polynomial.operations}")

            ds_ = [d for _, d in self.m.polynomial.operations]
            self.path_indices = sum((d.indices.tolist() for  d in ds_), [])
            self.path_coefficients = sum((d.stacked_coefficients.tolist() for d in ds_), [])
            self.num_segments_list = [ operand.num_segments for operand in self.descriptor.operands]
            #print(f"path_segment_indices len:{len(self.path_segment_indices)}, {self.path_segment_indices}")
            #print(f"len:{len(self.path_coefficients)}, path_coefficients:{self.path_coefficients}")

            path_segment_indices_tensor = torch.tensor(self.path_indices, dtype=torch.int32, device="cuda")
            path_coefficients_tensor = torch.tensor(self.path_coefficients, dtype=math_dtype, device="cuda")
            u_dim = list(self.descriptor.get_dims("u"))[0]
            w_seg_num, x_seg_num, y_seg_num, out_seg_num = self.num_segments_list[0], self.num_segments_list[1], self.num_segments_list[2], self.num_segments_list[3]
            i_list, j_list, k_list, v_list, coeff_list, v_offsets = build_grouped_paths(path_segment_indices_tensor, path_coefficients_tensor, out_seg_num, "cuda")
            self.u1d_meta = {}

            self.u1d_meta["i_list"] = i_list
            self.u1d_meta["j_list"] = j_list
            self.u1d_meta["k_list"] = k_list
            self.u1d_meta["v_list"] = v_list
            self.u1d_meta["coeff_list"] = coeff_list
            self.u1d_meta["v_offsets"] = v_offsets
            self.u1d_meta["out_seg_num"] = out_seg_num
            self.u1d_meta["w_seg_num"] = w_seg_num
            self.u1d_meta["x_seg_num"] = x_seg_num
            self.u1d_meta["y_seg_num"] = y_seg_num
            self.u1d_meta["u_dim"] = u_dim
        
        elif use_fasteq and (op_name == "fctp"):
            self.meta = infer_fctp_meta(self.descriptor, math_dtype=math_dtype, device="cuda")
            


    def __repr__(self):
        return self.repr + f"\n{super().__repr__()}"

    # For torch.jit.trace, we cannot pass explicit optionals,
    # so these must be passed as kwargs then.
    # List[Optional[Tensor]] does not work for similar reasons, hence, Dict
    # is the only option.
    # Also, shapes cannot be passed as integers, so they are passed via a
    # (potentially small-strided) tensor with the right shape.
    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """Compute the segmented polynomial based on the specified descriptor.

        Args:
            inputs: The input tensors. The number of input tensors must match
                the number of input buffers in the descriptor.
                Each input tensor should have a shape of ``(batch, operand_size)`` or
                ``(1, operand_size)`` or ``(index, operand_size)`` in the indexed case.
                Here, ``operand_size`` is the size of each operand as defined in
                the descriptor.
            input_indices: A dictionary that contains an optional indexing tensor
                for each input tensor. The key is the index into the inputs.
                If a key is not present, no indexing takes place.
                The contents of the index tensor must be suitable to index the
                input tensor (i.e., ``0 <= index_tensor[i] < input.shape[0]``).

                .. note::
                   Method ``"indexed_linear"`` requires the indices to be sorted.

            output_shapes: A dictionary specifying the size of the output batch
                dimensions using Tensors. We only read ``shape_tensor.shape[0]``.
                This is mandatory if the output tensor is indexed. Otherwise,
                the default shape is ``(batch, operand_size)``.
            output_indices: A dictionary that contains an optional indexing tensor
                for each output tensor. See ``input_indices`` for details.

        Returns:
            The output tensors resulting from the segmented polynomial.
            Their shapes are specified just like the inputs.
        """
        #print(f"op name:{self.op_name}, polynomial.operations:{self.polynomial.operations}")
        # General checks
        empty_dict: Dict[int, torch.Tensor] = {}
        if input_indices is None:
            input_indices = dict(empty_dict)
        if output_shapes is None:
            output_shapes = dict(empty_dict)
        if output_indices is None:
            output_indices = dict(empty_dict)

        inputs = list(inputs)

        if not torch.jit.is_scripting():
            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and not torch.fx._symbolic_trace.is_fx_tracing()
            ):
                torch._assert(
                    len(inputs) == self.num_inputs,
                    "the number of inputs must match the number of inputs of the polynomial",
                )

                for k, v in input_indices.items():
                    torch._assert(
                        0 <= k < self.num_inputs, "input index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_indices.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_shapes.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 2, "output shape must be two-dimensional")

                # If the input is on the CPU and we're using fused_tp, we need to fall back to naive
                if (
                    inputs[0].device == torch.device("cpu")
                    and self.method == "fused_tp"
                ):
                    warnings.warn(
                        "Fused TP is not supported on CPU. Falling back to naive implementation."
                    )
                    return self.fallback(
                        inputs, input_indices, output_shapes, output_indices
                    )

        if self.use_fasteq:
            out = [torch.empty(0) for _ in range(self.num_outputs)]
            if self.num_outputs != 1:
                    raise ValueError("equi_linear should have exactly one output")
            
            if self.op_name == "equi_linear":
                '''
                if tuple(inputs[0].shape) == (1, 36864) or tuple(inputs[0].shape) == (1, 163840) or tuple(inputs[0].shape) == (1, 852992):
                        torch.cuda.synchronize()
                        start_time = time.perf_counter() * 1000

                        ref = fast_equi_linear(self.descriptor, inputs[0], inputs[1])

                        torch.cuda.synchronize()
                        end_time = time.perf_counter() * 1000
                        execution_time_ms = end_time - start_time
                        print(f" fasteq equi-linear forward cost: {execution_time_ms:.3f} ms ")

                        torch.cuda.synchronize()
                        start_time = time.perf_counter() * 1000

                        out = self.m(inputs, input_indices, output_shapes, output_indices)
                        
                        torch.cuda.synchronize()
                        end_time = time.perf_counter() * 1000
                        execution_time_ms = end_time - start_time
                        print(f"cueq equi-linear forward cost: {execution_time_ms:.3f} ms")
                        print(f"eq-linear input0 shape: {inputs[0].shape}, input1 shape: {inputs[1].shape}")
                        print(f"eq-linear out shape:{out[0].shape}")
                '''
                if tuple(inputs[0].shape) == (1, 36864) :  # or tuple(inputs[0].shape) == (1, 163840) or tuple(inputs[0].shape) == (1, 852992)
                    ref = fast_equi_linear(self.descriptor, inputs[0], inputs[1])
                    out[0] = ref
                else:
                    out = self.m(inputs, input_indices, output_shapes, output_indices)

                    return out
            elif self.op_name == "stc":
                i0 = input_indices[0].to(torch.int32)
                x0 = inputs[0]
                x1 = inputs[1]
                
                x0 = x0.reshape(x0.shape[0], x0.shape[1] // self.u, self.u)
                x1 = x1.reshape(x1.shape[0], x1.shape[1] // self.u, self.u)

                #print(f"x1 shape:{x1.shape}, x0 shape:{x0.shape}, i0 shape:{i0.shape}")

                ref = fast_stc(
                    x1, x0, i0, 
                    self.coeffs_tensor, 
                    self.paths_tensor, 
                    self.path_lens_tensor, 
                    self.num_out_segments,
                )
                out[0] = ref
            elif self.op_name == "uniform1d" or (self.op_name == "cwtp" and self.u1d_compatible):
                
                w = inputs[0]
                x = inputs[1]
                y = inputs[2]
                scatter_sum_dim = x.shape[0]

                '''
                print(f"input:{input_indices[1]}")
                print(f"output:{output_indices[0]}")

                perm = torch.argsort(input_indices[1])
                input_sorted = input_indices[1][perm]
                out_sorted   = output_indices[0][perm]
                print(f"input sorted:{input_sorted}")
                print(f"output sorted:{out_sorted}")

                ib_list, icls_offsets = build_csr_buckets(input_indices[1], scatter_sum_dim)
                print(f"input sort b_list:{ib_list}")
                '''
                
                b_list, cls_offsets = build_csr_buckets(output_indices[0], scatter_sum_dim)

                #ref = fast_uniform1d_fused(w, x, y, input_indices[1], output_indices[0], b_list, cls_offsets, self.u1d_meta)
                ref = fast_uniform1d_jit(w, x, y, input_indices[1], output_indices[0], b_list, self.u1d_meta)
                ref = ref.view(scatter_sum_dim, -1)

                '''
                x_src = x[input_indices[1]]
                ref = fast_uniform1d(w, x_src, y, self.u1d_meta)
                ref = ref.view(x_src.shape[0], -1)
                ref = scatter_sum(ref, output_indices[0], dim=0, dim_size=scatter_sum_dim).view(scatter_sum_dim, -1)
                '''
                out[0] = ref
            elif self.op_name == "cwtp":
                # mptp case use input and output indices
                if input_indices.get(1) is not None and output_indices.get(0) is not None:
                    '''
                    for k, v in input_indices.items():
                        print(f"input_indices key:{k}, value:{v}")
                    for k, v in output_indices.items():
                        print(f"output_indices key:{k}, value:{v}")
                    for inp in inputs:
                        print(f"input shape:{inp.shape}")
                    '''
                    w, x, y = inputs[0], inputs[1], inputs[2]
                    sender = input_indices[1].to(torch.int32)
                    receiver= output_indices[0].to(torch.int32)
                    ref = fast_mptp(
                        w, x, y,sender, receiver,
                        self.meta,
                    )
                    out[0] = ref
                # cwtp case use only inputs
                else:
                    w, x, y = inputs[0], inputs[1], inputs[2]
                    ref = fast_cwtp(
                        w, x, y,
                        self.meta,
                        self.ell_meta,
                    )
                    out[0] = ref
            elif self.op_name == "fctp":
                w, x, y = inputs[0], inputs[1], inputs[2]
                ref = fast_fctp(
                    w, x, y,
                    self.meta,
                )
                out[0] = ref
            else:
                out = self.m(inputs, input_indices, output_shapes, output_indices)
                
        else:
            out = self.m(inputs, input_indices, output_shapes, output_indices)
        return out
