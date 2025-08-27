#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# collatz-search-perf-batched.py — max-perf batched packed-uint64 + Triton
#
# - Processes BATCH_SIZE (e.g., 10k) numbers in parallel.
# - All core operations (add, shift, ctz) are batched Triton kernels.
# - Autotuned for high performance on modern GPUs (e.g., RTX 3090).
# - Fused step logic with PyTorch masking to handle divergent paths (even/odd).
#
#   pip install torch triton

import os
import time
import logging
import numpy as np
from math import ceil

import torch
import triton
import triton.language as tl

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE      = 1024 * 64
BEST_ORIGINALS_POOL_SIZE = 1024
num_elite = BATCH_SIZE // 4
num_random = BATCH_SIZE // 4

BITS            = 1024 # * 2
WORD_BITS       = 64
WORDS_TARGET    = (BITS + WORD_BITS - 1) // WORD_BITS # 157
# EXTRA_HEADROOM  = 3
EXTRA_HEADROOM  = WORDS_TARGET * 3
_words_needed   = WORDS_TARGET + EXTRA_HEADROOM       # 160
# Pad to next power of 2 for tl.arange in k_find_hi_batched
WORDS_CAP       = 1 << (_words_needed - 1).bit_length() if _words_needed > 0 else 1 # e.g. 160 -> 256
# Max blocks needed for carry propagation across WORDS_CAP
MAX_BLOCKS      = (WORDS_CAP + 32 - 1) // 32 # Max blocks assuming smallest block size for tuning
# Tuned for 10k-bit numbers on RTX 3090. Autotuning BLOCK_WORDS is complex
# because CPU-side logic depends on it. A fixed value is used instead.
# For different hardware or bit lengths, manually tuning this is recommended.
BLOCK_WORDS     = 128
DEVICE          = "cuda"
DTYPE           = torch.uint64
UINT64_MAX      = (1 << 64) - 1

# ----------------------------
# High-Performance Batched Kernels
# ----------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['n_words_op'],
    reset_to_zero=['out_ptr', 'carry0_ptr', 'allones_ptr'],
)
@triton.jit
def k_add_block_pass1_batched(
    a_ptr, b_ptr, out_ptr, carry0_ptr, allones_ptr,
    n_words_op, n_blocks,
    WORDS_STRIDE: tl.constexpr, CARRY_STRIDE: tl.constexpr,
    BLOCK_WORDS: tl.constexpr):

    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    base_a = a_ptr + pid_batch * WORDS_STRIDE
    base_b = b_ptr + pid_batch * WORDS_STRIDE
    base_out = out_ptr + pid_batch * WORDS_STRIDE
    base_carry0 = carry0_ptr + pid_batch * CARRY_STRIDE
    base_allones = allones_ptr + pid_batch * CARRY_STRIDE

    start = pid_block * BLOCK_WORDS
    limit = tl.minimum(n_words_op, start + BLOCK_WORDS)

    carry = tl.zeros((), dtype=tl.uint64)
    allones = tl.full((), 1, tl.int1)

    i = start
    while i < limit:
        ai = tl.load(base_a + i, mask=i < n_words_op, other=0)
        bi = tl.load(base_b + i, mask=i < n_words_op, other=0)
        tmp = ai + bi
        s = tmp + carry
        c1 = tmp < ai
        c2 = s < tmp
        carry = tl.cast(tl.where(c1 | c2, 1, 0), tl.uint64)
        tl.store(base_out + i, s, mask=i < n_words_op)
        allones = allones & (s == 0xFFFFFFFFFFFFFFFF)
        i += 1

    if start < n_words_op:
        tl.store(base_carry0 + pid_block, tl.cast(carry, tl.uint8))
        tl.store(base_allones + pid_block, tl.cast(allones, tl.uint8))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=[],
    reset_to_zero=['c_in_ptr'],
)
@triton.jit
def k_propagate_carries_scan_batched(
    carry0_ptr, allones_ptr, c_in_ptr, n_blocks_ptr, initial_carry,
    CARRY_STRIDE: tl.constexpr):

    pid_batch = tl.program_id(0)

    base_carry0 = carry0_ptr + pid_batch * CARRY_STRIDE
    base_allones = allones_ptr + pid_batch * CARRY_STRIDE
    base_c_in = c_in_ptr + pid_batch * (CARRY_STRIDE + 1)

    n_blocks = tl.load(n_blocks_ptr + pid_batch)

    carry = tl.cast(initial_carry, tl.uint8)
    i = 0
    while i < n_blocks:
        tl.store(base_c_in + i, carry)
        c0 = tl.load(base_carry0 + i)
        a1 = tl.load(base_allones + i)
        carry = c0 | (a1 & carry)
        i += 1
    tl.store(base_c_in + n_blocks, carry)

@triton.jit
def k_inc_blocks_pass2_batched(
    out_ptr, c_in_ptr, n_words_op,
    WORDS_STRIDE: tl.constexpr, CARRY_STRIDE: tl.constexpr,
    BLOCK_WORDS: tl.constexpr):

    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)

    base_out = out_ptr + pid_batch * WORDS_STRIDE
    base_c_in = c_in_ptr + pid_batch * (CARRY_STRIDE + 1)

    start = pid_block * BLOCK_WORDS
    if start >= n_words_op: return

    limit = tl.minimum(n_words_op, start + BLOCK_WORDS)
    cin = tl.load(base_c_in + pid_block)
    carry = tl.where(cin != 0, 1, 0)

    i = start
    while i < limit:
        s = tl.load(base_out + i)
        s2 = s + carry
        overflow = (carry == 1) & (s == 0xFFFFFFFFFFFFFFFF)
        w = tl.where(carry == 1, s2, s)
        tl.store(base_out + i, w)
        carry = tl.where(overflow, 1, 0)
        i += 1

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=[],
)
@triton.jit
def k_shl1_batched(in_ptr, out_ptr, n_words_ptr, WORDS_STRIDE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_word = tl.program_id(1)

    base_in = in_ptr + pid_batch * WORDS_STRIDE
    base_out = out_ptr + pid_batch * WORDS_STRIDE
    n_words = tl.load(n_words_ptr + pid_batch)

    if pid_word >= n_words: return

    cur = tl.load(base_in + pid_word)
    prev = tl.load(base_in + pid_word - 1, mask=pid_word > 0, other=0)
    carry_in = prev >> 63
    out = (cur << 1) | carry_in
    tl.store(base_out + pid_word, out)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=[],
)
@triton.jit
def k_shr_any_batched(in_ptr, out_ptr, n_words_ptr, k_bits_ptr, WORDS_STRIDE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_word_out = tl.program_id(1)

    base_in = in_ptr + pid_batch * WORDS_STRIDE
    base_out = out_ptr + pid_batch * WORDS_STRIDE
    n_words = tl.load(n_words_ptr + pid_batch)
    k_bits = tl.load(k_bits_ptr + pid_batch)

    if pid_word_out >= WORDS_STRIDE: return

    word_shift = k_bits // 64
    bit_shift = k_bits % 64

    src_idx = pid_word_out + word_shift
    mask_low = src_idx < n_words
    mask_high = (src_idx + 1) < n_words

    low = tl.load(base_in + src_idx, mask=mask_low, other=0)
    if bit_shift == 0:
        out = low
    else:
        hi = tl.load(base_in + src_idx + 1, mask=mask_high, other=0)
        out = (low >> bit_shift) | (hi << (64 - bit_shift))
    tl.store(base_out + pid_word_out, out)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=[],
    reset_to_zero=['hi_out_ptr'],
)
@triton.jit
def k_find_hi_batched(words_ptr, hi_out_ptr, WORDS_STRIDE: tl.constexpr):
    pid_batch = tl.program_id(0)
    base_words = words_ptr + pid_batch * WORDS_STRIDE
    indices = tl.arange(0, WORDS_STRIDE)
    words = tl.load(base_words + indices)
    is_nonzero = words != 0
    indices_or_sentinel = tl.where(is_nonzero, indices, -1)
    hi = tl.max(indices_or_sentinel, axis=0)
    tl.store(hi_out_ptr + pid_batch, hi)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def k_find_first_nonzero_word_batched(words_ptr, n_words_ptr, result_idx_ptr,
                                      WORDS_STRIDE: tl.constexpr,
                                      BLOCK_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    tid = tl.program_id(1)

    base_ptr = words_ptr + pid_batch * WORDS_STRIDE
    n_words = tl.load(n_words_ptr + pid_batch)

    idx = tid
    while idx < n_words:
        w = tl.load(base_ptr + idx, mask=idx < n_words, other=0)
        if w != 0:
            tl.atomic_min(result_idx_ptr + pid_batch, idx)
        idx += BLOCK_SIZE

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=[],
    reset_to_zero=['result_k_ptr'],
)
@triton.jit
def k_word_ctz_batched(words_ptr, word_idx_ptr, result_k_ptr, WORDS_STRIDE: tl.constexpr):
    pid_batch = tl.program_id(0)

    base_ptr = words_ptr + pid_batch * WORDS_STRIDE
    idx = tl.load(word_idx_ptr + pid_batch)

    word = tl.load(base_ptr + idx, mask=idx != -1, other=0)

    bit_pos = tl.arange(0, 64)
    is_set = ((word >> bit_pos) & 1) != 0
    pos_or_64 = tl.where(is_set, bit_pos, 64)
    word_ctz = tl.min(pos_or_64)

    total_ctz = (tl.cast(idx, tl.int64) * 64) + tl.cast(word_ctz, tl.int64)
    final_ctz = tl.where(idx == -1, (tl.cast(WORDS_STRIDE, tl.int64)) * 64, total_ctz)
    tl.store(result_k_ptr + pid_batch, final_ctz)

# ----------------------------
# Batched Core Ops
# ----------------------------
@torch.no_grad()
def find_hi_batched(words, out_hi):
    grid = (words.shape[0],)
    k_find_hi_batched[grid](words, out_hi, WORDS_STRIDE=WORDS_CAP)

@torch.no_grad()
def shl1_batched(x_words, x_hi, out_words, out_hi, scratch):
    out_words.zero_()
    max_hi = int(x_hi.max().item())
    n_words_op = min(max_hi + 2, WORDS_CAP)
    if n_words_op <= 0:
        out_hi.fill_(-1)
        return torch.zeros(x_words.shape[0], dtype=torch.bool, device=DEVICE)

    n_words_in = x_hi + 1
    grid = (x_words.shape[0], n_words_op)
    k_shl1_batched[grid](x_words, out_words, n_words_in, WORDS_STRIDE=WORDS_CAP)

    top_words_indices = torch.clamp(x_hi, min=0)
    top_words = x_words.to(torch.int64)[torch.arange(x_words.shape[0]), top_words_indices]
    top_carry = (top_words < 0).to(DTYPE)
    has_space = x_hi + 1 < WORDS_CAP

    mask = (top_carry == 1) & has_space & (x_hi >= 0)
    new_hi_indices = x_hi[mask] + 1
    if mask.any():
        out_words.view(torch.int64)[mask, new_hi_indices] = 1

    find_hi_batched(out_words, out_hi)

    overflow_mask = (top_carry == 1) & ~has_space & (x_hi >= 0)
    return overflow_mask

@torch.no_grad()
def shr_k_inplace_batched(x_words, x_hi, k_bits, tmp_buf):
    n_words_in = x_hi + 1
    grid = (x_words.shape[0], WORDS_CAP)
    k_shr_any_batched[grid](x_words, tmp_buf, n_words_in, k_bits, WORDS_STRIDE=WORDS_CAP)
    x_words.copy_(tmp_buf)
    find_hi_batched(x_words, x_hi)

@torch.no_grad()
def add_batched(a_words, a_hi, b_words, b_hi, out_words, out_hi, scratch, plus_one=False):
    out_words.zero_()
    max_hi_ab = torch.maximum(a_hi, b_hi)

    n_words_op = int(max_hi_ab.max().item()) + 2
    n_words_op = min(n_words_op, WORDS_CAP)
    if n_words_op < 1: n_words_op = 1

    n_blocks = (n_words_op + BLOCK_WORDS - 1) // BLOCK_WORDS
    n_blocks_per_num = torch.clamp(((max_hi_ab + 2 + BLOCK_WORDS - 1) // BLOCK_WORDS), min=1, max=MAX_BLOCKS).to(torch.int32)

    carry0, allones, c_in = scratch['carry0'], scratch['allones'], scratch['c_in']
    c_in.zero_() # FIX: prevent reading stale carry data for smaller numbers in batch

    grid1 = (a_words.shape[0], n_blocks)
    k_add_block_pass1_batched[grid1](a_words, b_words, out_words, carry0, allones,
        n_words_op, n_blocks, WORDS_STRIDE=WORDS_CAP, CARRY_STRIDE=MAX_BLOCKS, BLOCK_WORDS=BLOCK_WORDS)

    grid1_5 = (a_words.shape[0],)
    k_propagate_carries_scan_batched[grid1_5](carry0, allones, c_in, n_blocks_per_num,
        initial_carry=1 if plus_one else 0, CARRY_STRIDE=MAX_BLOCKS)

    grid2 = (a_words.shape[0], n_blocks)
    k_inc_blocks_pass2_batched[grid2](out_words, c_in, n_words_op,
        WORDS_STRIDE=WORDS_CAP, CARRY_STRIDE=MAX_BLOCKS, BLOCK_WORDS=BLOCK_WORDS)

    final_carry_idx = n_blocks_per_num
    final_carry = c_in[torch.arange(a_words.shape[0]), final_carry_idx]

    final_carry_word_idx = max_hi_ab + 1
    mask = (final_carry == 1) & (final_carry_word_idx < WORDS_CAP)
    if mask.any():
        indices = final_carry_word_idx[mask]
        out_words.view(torch.int64)[mask, indices] = 1

    find_hi_batched(out_words, out_hi)

    overflow_mask = (final_carry == 1) & (final_carry_word_idx >= WORDS_CAP)
    return overflow_mask

@torch.no_grad()
def ctz_batched(x_words, x_hi, scratch):
    n_words = torch.clamp(x_hi + 1, min=0)
    result_idx, result_k = scratch['ctz_result_idx'], scratch['ctz_result_k']
    result_idx.fill_(WORDS_CAP)

    grid1 = lambda META: (x_words.shape[0], META['BLOCK_SIZE'])
    k_find_first_nonzero_word_batched[grid1](
        x_words, n_words, result_idx, WORDS_STRIDE=WORDS_CAP)

    word_idx = torch.where(result_idx >= n_words, -1, result_idx)
    grid2 = (x_words.shape[0],)
    k_word_ctz_batched[grid2](x_words, word_idx, result_k, WORDS_STRIDE=WORDS_CAP)
    return result_k

# ----------------------------
# Fused Collatz Step Logic (Batched)
# ----------------------------
@torch.no_grad()
def is_even_batched(x_words):
    return (x_words[:, 0].to(torch.int64) % 2) == 0

@torch.no_grad()
def terminated_leq_8_batched(x_words, x_hi):
    is_small = x_hi <= 0
    val_leq_8 = x_words[:, 0].to(torch.int64) <= 8
    return is_small & val_leq_8

@torch.no_grad()
def do_steps_fused_batched(x_words, x_hi, scratch, num_steps, active_mask):
    steps_tensor = torch.zeros(x_words.shape[0], dtype=torch.int64, device=DEVICE)
    tmp_words, tmp_hi = scratch['tmp_words'], scratch['tmp_hi']
    out_words, out_hi = scratch['out_words'], scratch['out_hi']

    for i in range(num_steps):
        term_mask = terminated_leq_8_batched(x_words, x_hi)
        current_active = active_mask & ~term_mask
        if not current_active.any(): break

        is_even = is_even_batched(x_words)
        odd_mask = current_active & ~is_even

        shl_overflow_mask = shl1_batched(x_words, x_hi, tmp_words, tmp_hi, scratch)
        add_overflow_mask = add_batched(x_words, x_hi, tmp_words, tmp_hi, out_words, out_hi, scratch, plus_one=True)
        step_overflow_mask = (shl_overflow_mask | add_overflow_mask) & odd_mask
        if step_overflow_mask.any():
            idx = torch.where(step_overflow_mask)[0][0].item()
            # Use a local function to avoid circular dependency with test utils
            def _num_to_hex(w, h):
                val = 0
                for i in range(h, -1, -1): val = (val << 64) | int(w[i])
                return hex(val)
            num_str = _num_to_hex(x_words[idx].cpu().numpy(), int(x_hi[idx].item()))
            raise RuntimeError(f"FATAL: number overflowed at index {idx}. Number: {num_str[:40]}...")

        k_odd = ctz_batched(out_words, out_hi, scratch).clone() # clone is crucial
        shr_k_inplace_batched(out_words, out_hi, k_odd, scratch['shr_tmp'])

        k_even = ctz_batched(x_words, x_hi, scratch)
        shr_k_inplace_batched(tmp_words.copy_(x_words), tmp_hi.copy_(x_hi), k_even, scratch['shr_tmp'])

        next_words = torch.where(odd_mask.view(-1, 1), out_words.view(torch.int64), tmp_words.view(torch.int64)).view(DTYPE)
        next_hi = torch.where(odd_mask, out_hi, tmp_hi)

        x_words.copy_(torch.where(current_active.view(-1, 1), next_words.view(torch.int64), x_words.view(torch.int64)).view(DTYPE))
        x_hi.copy_(torch.where(current_active, next_hi, x_hi))

        steps_this_iter = torch.where(odd_mask, 1 + k_odd, k_even)
        steps_tensor.add_(torch.where(current_active, steps_this_iter, 0))

    return steps_tensor, active_mask & ~terminated_leq_8_batched(x_words, x_hi)

# ----------------------------
# Evolutionary Search Logic
# ----------------------------
@torch.no_grad()
def tournament_selection(population_words, fitness, num_selections, tournament_size=3):
    population_size = population_words.shape[0]
    contender_indices = torch.randint(0, population_size, (num_selections, tournament_size), device=DEVICE)
    contender_fitness = fitness[contender_indices]
    winner_indices_in_tournaments = torch.argmax(contender_fitness, dim=1)
    final_indices = contender_indices[torch.arange(num_selections), winner_indices_in_tournaments]
    return population_words.view(torch.int64)[final_indices].view(DTYPE)

@torch.no_grad()
def breed(p1_words, p2_words, bits_to_flip: int):
    num_children = p1_words.shape[0]
    device = p1_words.device
    if num_children == 0:
        return torch.empty((0, WORDS_CAP), dtype=DTYPE, device=device)

    # Crossover
    crossover_point = torch.randint(1, WORDS_TARGET, (num_children, 1), device=device)
    mask = torch.arange(WORDS_CAP, device=device) < crossover_point
    children_words_i64 = torch.where(mask, p1_words.view(torch.int64), p2_words.view(torch.int64))

    # Mutation
    bit_indices = torch.randint(0, BITS, (num_children, bits_to_flip), device=device)
    word_indices = bit_indices // WORD_BITS
    bit_in_word_indices = bit_indices % WORD_BITS

    xor_masks_vals_i64 = torch.tensor(1, dtype=torch.int64, device=device) << bit_in_word_indices

    row_indices = torch.arange(num_children, device=device)
    for i in range(bits_to_flip):
        word_idx_for_col_i = word_indices[:, i]
        bit_mask_for_col_i_i64 = xor_masks_vals_i64[:, i]
        words_to_mutate_i64 = children_words_i64[row_indices, word_idx_for_col_i]
        mutated_segment_i64 = words_to_mutate_i64 ^ bit_mask_for_col_i_i64
        children_words_i64[row_indices, word_idx_for_col_i] = mutated_segment_i64

    first_words_i64 = children_words_i64[:, 0] | 1
    children_words_i64[:, 0] = first_words_i64

    return children_words_i64.view(DTYPE)

# ----------------------------
# Random init & Test Utils
# ----------------------------
@torch.no_grad()
def random_big_batched(seed_ns, n=None):
    if n is None: n = BATCH_SIZE
    rng = np.random.default_rng(seed_ns)
    shape = (n, WORDS_CAP)
    words_np = rng.integers(0, 1 << 32, size=shape, dtype=np.uint32).astype(np.uint64)
    words_np |= rng.integers(0, 1 << 32, size=shape, dtype=np.uint32).astype(np.uint64) << 32
    words_np[:, 0] |= 9
    words_np[:, WORDS_TARGET - 1] |= (np.uint64(1) << np.uint64((BITS - 1) % 64))
    if WORDS_TARGET < WORDS_CAP: words_np[:, WORDS_TARGET:] = 0
    words = torch.from_numpy(words_np).to(DEVICE)
    hi = torch.empty(n, dtype=torch.int32, device=DEVICE)
    find_hi_batched(words, hi)
    return words, hi

def big_to_int(words_cpu, hi):
    acc = 0
    for i in range(hi, -1, -1):
        acc = (acc << 64) | int(words_cpu[i])
    return acc

def int_to_big(n, cap=WORDS_CAP):
    words_list = []
    while n > 0:
        words_list.append(n & UINT64_MAX)
        n >>= 64
    words_np = np.array(words_list, dtype=np.uint64)
    final_words = np.zeros(cap, dtype=np.uint64)
    num_to_copy = min(len(words_np), cap)
    if num_to_copy > 0: final_words[:num_to_copy] = words_np[:num_to_copy]
    hi = -1
    for i in range(len(final_words) - 1, -1, -1):
        if final_words[i] != 0:
            hi = i
            break
    return final_words, hi

def collatz_step_python(n):
    if n <= 8 and n > 0: return n, 0
    if n % 2 == 0:
        k = (n & -n).bit_length() - 1 if n != 0 else 0
        return n >> k, k
    else:
        n = 3 * n + 1
        k = (n & -n).bit_length() - 1 if n != 0 else 0
        return n >> k, 1 + k

@torch.no_grad()
def run_tests():
    print("[tests] starting…")
    global BATCH_SIZE
    orig_batch_size, BATCH_SIZE = BATCH_SIZE, 8
    scratch = make_scratch_pad(BATCH_SIZE)
    BIT_MASK = (1 << (64 * WORDS_CAP)) - 1

    def _assert_normalized_b(w, h, tag):
        h_check = torch.empty_like(h)
        find_hi_batched(w, h_check)
        assert torch.equal(h, h_check), f"Normalization fail for {tag}: hi={h.cpu().numpy()} vs expected={h_check.cpu().numpy()}"

    def _ints_to_batch(py_ints, cap=WORDS_CAP):
        words = [int_to_big(n, cap)[0] for n in py_ints]
        w = torch.from_numpy(np.stack(words)).to(DEVICE)
        h = torch.empty(len(py_ints), dtype=torch.int32, device=DEVICE)
        find_hi_batched(w, h)
        return w, h

    # 1. Add
    a_ints = [int.from_bytes(os.urandom(WORDS_CAP * 4), "little") for _ in range(BATCH_SIZE)]
    b_ints = [int.from_bytes(os.urandom(WORDS_CAP * 4), "little") for _ in range(BATCH_SIZE)]
    aw, ah = _ints_to_batch(a_ints)
    bw, bh = _ints_to_batch(b_ints)
    ow, oh = scratch['out_words'], scratch['out_hi']
    _ = add_batched(aw, ah, bw, bh, ow, oh, scratch, plus_one=True)
    _assert_normalized_b(ow, oh, "add")
    for i in range(BATCH_SIZE):
        expect = (a_ints[i] + b_ints[i] + 1) & BIT_MASK
        got = big_to_int(ow[i].cpu().numpy(), int(oh[i].item()))
        assert got == expect, "add mismatch"
    print("[tests] add_batched OK")

    # 2. CTZ
    k_vals = [0, 1, 63, 64, 65, 127, 128, 511]
    ctz_ints = [1 << k for k in k_vals]
    cw, ch = _ints_to_batch(ctz_ints)
    k_gpu = ctz_batched(cw, ch, scratch)
    assert k_gpu.cpu().numpy().tolist() == k_vals, "ctz mismatch"
    print("[tests] ctz_batched OK")

    # 3. Shifts
    s_ints = [int.from_bytes(os.urandom(WORDS_CAP * 4), 'little') for _ in range(BATCH_SIZE)]
    sw, sh = _ints_to_batch(s_ints)
    shl_ow, shl_oh = scratch['out_words'], scratch['out_hi']
    _ = shl1_batched(sw, sh, shl_ow, shl_oh, scratch)
    _assert_normalized_b(shl_ow, shl_oh, "shl1")
    for i in range(BATCH_SIZE):
        expect = (s_ints[i] << 1) & BIT_MASK
        got = big_to_int(shl_ow[i].cpu().numpy(), int(shl_oh[i].item()))
        assert got == expect, "shl1 mismatch"

    k_shr = torch.randint(0, 64 * WORDS_CAP, (BATCH_SIZE,), device=DEVICE, dtype=torch.int64)
    shr_w, shr_h = sw.clone(), sh.clone()
    shr_k_inplace_batched(shr_w, shr_h, k_shr, scratch['shr_tmp'])
    _assert_normalized_b(shr_w, shr_h, "shr")
    for i in range(BATCH_SIZE):
        expect = s_ints[i] >> int(k_shr[i].item())
        got = big_to_int(shr_w[i].cpu().numpy(), int(shr_h[i].item()))
        assert got == expect, f"shr mismatch k={k_shr[i]}"
    print("[tests] shifts_batched OK")

    # 4. E2E step
    xw, xh = random_big_batched(12345)
    py_nums = [big_to_int(xw[i].cpu().numpy(), int(xh[i].item())) for i in range(BATCH_SIZE)]
    py_next = [collatz_step_python(n)[0] for n in py_nums]
    py_steps = [collatz_step_python(n)[1] for n in py_nums]
    active = torch.ones(BATCH_SIZE, dtype=torch.bool, device=DEVICE)
    gpu_steps, _ = do_steps_fused_batched(xw, xh, scratch, 1, active)
    gpu_next = [big_to_int(xw[i].cpu().numpy(), int(xh[i].item())) for i in range(BATCH_SIZE)]
    assert py_next == gpu_next, f"E2E val mismatch:\nPy: {py_next}\nGPU:{gpu_next}"
    assert py_steps == gpu_steps.cpu().numpy().tolist(), f"E2E steps mismatch"
    print("[tests] single fused step OK")

    BATCH_SIZE = orig_batch_size
    print("[tests] All OK")


@torch.no_grad()
def run_extra_tests():
    print("[tests-extra] starting…")
    global BATCH_SIZE
    orig_batch_size, BATCH_SIZE = BATCH_SIZE, 8
    scratch = make_scratch_pad(BATCH_SIZE)
    BIT_MASK = (1 << (64 * WORDS_CAP)) - 1

    def _assert_normalized(w, h, tag):
        h_check = torch.empty_like(h)
        find_hi_batched(w, h_check)
        assert torch.equal(h, h_check), f"[{tag}] hi mismatch"
        # words above hi must be zero
        for i in range(w.shape[0]):
            hi = int(h[i].item())
            if hi + 1 < WORDS_CAP:
                assert (w[i, hi+1:] == 0).all(), f"[{tag}] nonzero garbage above hi for row {i}"

    def _ints_to_batch(py_ints, cap=WORDS_CAP):
        words = [int_to_big(n, cap)[0] for n in py_ints]
        w = torch.from_numpy(np.stack(words)).to(DEVICE)
        h = torch.empty(len(py_ints), dtype=torch.int32, device=DEVICE)
        find_hi_batched(w, h)
        return w, h

    def _big_to_int_row(wrow, hi):
        return big_to_int(wrow.cpu().numpy(), int(hi.item()))

    # 1) ADD: carry across a block boundary (no +1)
    # a = (2^(64*BLOCK_WORDS)) - 1  (all ones in lower BLOCK_WORDS words)
    # b = 1                         (forces carry across entire lower block)
    a1 = (1 << (64 * BLOCK_WORDS)) - 1
    b1 = 1
    aw, ah = _ints_to_batch([a1] * BATCH_SIZE)
    bw, bh = _ints_to_batch([b1] * BATCH_SIZE)
    ow, oh = scratch['out_words'], scratch['out_hi']
    _ = add_batched(aw, ah, bw, bh, ow, oh, scratch, plus_one=False)
    _assert_normalized(ow, oh, "add-carry-block")
    for i in range(BATCH_SIZE):
        got = _big_to_int_row(ow[i], oh[i])
        expect = (a1 + b1) & BIT_MASK
        assert got == expect, "[add-carry-block] mismatch"

    # 2) ADD: propagate +1 through an all-ones block
    # sum = (2^(64*BLOCK_WORDS)) - 1; then +1 should produce 2^(64*BLOCK_WORDS)
    aw, ah = _ints_to_batch([ (1 << (64 * BLOCK_WORDS)) - 1 ] * BATCH_SIZE)
    bw, bh = _ints_to_batch([0] * BATCH_SIZE)
    _ = add_batched(aw, ah, bw, bh, ow, oh, scratch, plus_one=True)
    _assert_normalized(ow, oh, "add-plusone-propagate")
    for i in range(BATCH_SIZE):
        got = _big_to_int_row(ow[i], oh[i])
        expect = (1 << (64 * BLOCK_WORDS)) & BIT_MASK
        assert got == expect, "[add-plusone-propagate] mismatch"

    # 3) ADD: overflow detection near mask limit
    a2 = BIT_MASK - 5
    b2 = 10
    aw, ah = _ints_to_batch([a2] * BATCH_SIZE)
    bw, bh = _ints_to_batch([b2] * BATCH_SIZE)
    overflow_mask = add_batched(aw, ah, bw, bh, ow, oh, scratch, plus_one=False)
    assert overflow_mask.all(), "[add-overflow] failed to detect overflow"

    # 4) SHL1: top-carry should bump hi by 1 and set new bit0
    k = 10  # any safe index (< WORDS_CAP-2)
    val = 1 << (64 * k + 63)  # MSB of word k
    sw, sh = _ints_to_batch([val] * BATCH_SIZE)
    shl_ow, shl_oh = scratch['out_words'], scratch['out_hi']
    _ = shl1_batched(sw, sh, shl_ow, shl_oh, scratch)
    _assert_normalized(shl_ow, shl_oh, "shl1-top-carry")
    for i in range(BATCH_SIZE):
        got = _big_to_int_row(shl_ow[i], shl_oh[i])
        expect = (val << 1) & BIT_MASK
        assert got == expect, "[shl1-top-carry] mismatch"

    # 5) SHR extremes: k = 0 and k = (64*WORDS_CAP - 1)
    s_ints = [int.from_bytes(os.urandom(WORDS_CAP * 4), 'little') for _ in range(BATCH_SIZE)]
    sw, sh = _ints_to_batch(s_ints)
    # k = 0
    shr_w0, shr_h0 = sw.clone(), sh.clone()
    k0 = torch.zeros(BATCH_SIZE, dtype=torch.int64, device=DEVICE)
    shr_k_inplace_batched(shr_w0, shr_h0, k0, scratch['shr_tmp'])
    _assert_normalized(shr_w0, shr_h0, "shr-k0")
    for i in range(BATCH_SIZE):
        assert _big_to_int_row(shr_w0[i], shr_h0[i]) == s_ints[i], "[shr-k0] mismatch"
    # k = max-1
    shr_wm, shr_hm = sw.clone(), sh.clone()
    kmaxm1 = torch.full((BATCH_SIZE,), 64 * WORDS_CAP - 1, dtype=torch.int64, device=DEVICE)
    shr_k_inplace_batched(shr_wm, shr_hm, kmaxm1, scratch['shr_tmp'])
    _assert_normalized(shr_wm, shr_hm, "shr-kmax-1")
    for i in range(BATCH_SIZE):
        expect = s_ints[i] >> (64 * WORDS_CAP - 1)
        got = _big_to_int_row(shr_wm[i], shr_hm[i])
        assert got == expect, "[shr-kmax-1] mismatch"

    # 6) CTZ(0) sentinel
    cw, ch = _ints_to_batch([0] * BATCH_SIZE)
    k_gpu = ctz_batched(cw, ch, scratch)
    sentinel = WORDS_CAP * 64
    assert (k_gpu == sentinel).all(), "[ctz-zero] sentinel mismatch"

    # 7) Odd-path step counts where 3n+1 has many trailing zeros
    test_ns = [5, 21, 85, 341] * (BATCH_SIZE // 4)  # 3n+1 = 16, 64, 256, 1024
    xw, xh = _ints_to_batch(test_ns)
    active = torch.ones(len(test_ns), dtype=torch.bool, device=DEVICE)
    steps, _ = do_steps_fused_batched(xw, xh, scratch, 1, active)
    next_vals = [ _big_to_int_row(xw[i], xh[i]) for i in range(len(test_ns)) ]
    py_next = []
    py_steps = []
    for n in test_ns:
        nn, kk = collatz_step_python(n)
        py_next.append(nn); py_steps.append(kk)
    assert next_vals == py_next, f"[odd-path] next mismatch: {next_vals} vs {py_next}"
    assert steps.cpu().tolist() == py_steps, f"[odd-path] step mismatch: {steps} vs {py_steps}"

    # 8) Overflow detection
    # Craft a number that is guaranteed to overflow on 3n+1
    overflow_n = (1 << (64 * WORDS_CAP)) - 1
    # Use a BATCH_SIZE of 1 for this test
    xw_ovf, xh_ovf = _ints_to_batch([overflow_n] * 1)
    active_ovf = torch.ones(1, dtype=torch.bool, device=DEVICE)
    try:
        do_steps_fused_batched(xw_ovf, xh_ovf, scratch, 1, active_ovf)
        assert False, "[overflow-test] did not raise RuntimeError on overflow"
    except RuntimeError as e:
        if "overflowed" in str(e):
            print("[tests-extra] Overflow check OK")
        else:
            assert False, f"[overflow-test] raised unexpected RuntimeError: {e}"
    except Exception as e:
        assert False, f"[overflow-test] raised unexpected exception type: {type(e).__name__}: {e}"

    print("[tests-extra] All OK")
    BATCH_SIZE = orig_batch_size

# ----------------------------
# Main
# ----------------------------
def make_scratch_pad(batch_size):
    return {
        'carry0': torch.empty(batch_size, MAX_BLOCKS, dtype=torch.uint8, device=DEVICE),
        'allones': torch.empty(batch_size, MAX_BLOCKS, dtype=torch.uint8, device=DEVICE),
        'c_in': torch.empty(batch_size, MAX_BLOCKS + 1, dtype=torch.uint8, device=DEVICE),
        'shr_tmp': torch.empty(batch_size, WORDS_CAP, dtype=DTYPE, device=DEVICE),
        'ctz_result_idx': torch.empty(batch_size, dtype=torch.int32, device=DEVICE),
        'ctz_result_k': torch.empty(batch_size, dtype=torch.int64, device=DEVICE),
        'tmp_words': torch.empty(batch_size, WORDS_CAP, dtype=DTYPE, device=DEVICE),
        'tmp_hi': torch.empty(batch_size, dtype=torch.int32, device=DEVICE),
        'out_words': torch.empty(batch_size, WORDS_CAP, dtype=DTYPE, device=DEVICE),
        'out_hi': torch.empty(batch_size, dtype=torch.int32, device=DEVICE),
    }

def main():
    assert torch.cuda.is_available(), "CUDA required."
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")


    scratch_pad = make_scratch_pad(BATCH_SIZE)
    logging.info(f"Starting main loop: BATCH_SIZE={BATCH_SIZE}, BITS={BITS}")
    total_steps_all_time, start_time_all_time = 0, time.time()
    record_steps, record_idx, record_gen = 0, 0, 0

    best_originals_words = torch.empty((0, WORDS_CAP), dtype=DTYPE, device=DEVICE)
    best_originals_fitness = torch.empty(0, dtype=torch.int64, device=DEVICE)

    logging.info("Generating initial random population...")
    x_words, x_hi = random_big_batched(time.time_ns())
    is_original_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device=DEVICE)
    generation_num = 0
    if not os.path.exists("batches"): os.makedirs("batches")
    if not os.path.exists("records"): os.makedirs("records")

    ran_tests = False
    while True:
        generation_num += 1
        logging.info(f"Starting generation {generation_num}...")

        torch.save({'words': x_words.cpu()}, f"batches/batch_{generation_num:04d}.pt")
        initial_x_words_this_gen = x_words.clone()
        initial_is_original_mask_this_gen = is_original_mask.clone()

        total_steps_this_batch = torch.zeros(BATCH_SIZE, dtype=torch.int64, device=DEVICE)
        active_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device=DEVICE)
        outer_loop_start, chunk_num = time.time(), 0

        while active_mask.any():
            chunk_num += 1
            num_active_pre = active_mask.sum().item()
            if num_active_pre > 1:
                first_active_idx = torch.where(active_mask)[0][0]
                first_active_row = x_words[first_active_idx]
                equality_matrix = (x_words.view(torch.int64) == first_active_row.view(torch.int64))
                rows_match_first = equality_matrix.all(dim=1)
                if rows_match_first[active_mask].all():
                    logging.warning(f"!!! gen {generation_num}: chunk {chunk_num}, all {num_active_pre} active numbers are identical!")

            chunk_steps, active_mask = do_steps_fused_batched(x_words, x_hi, scratch_pad, 10000, active_mask)
            total_steps_this_batch.add_(chunk_steps)

            num_active = active_mask.sum().item()
            total_batch_steps = total_steps_this_batch.sum().item()
            logging.info(f"gen {generation_num}: chunk {chunk_num}: processed 10k steps. "
                         f"{num_active}/{BATCH_SIZE} active. Total steps: {total_batch_steps}")

            if num_active > 0 and chunk_steps[active_mask].sum() == 0:
                logging.warning(f"gen {generation_num}: zero-progress chunk on {num_active} active; reseeding population")
                x_words, x_hi = random_big_batched(time.time_ns())
                is_original_mask.fill_(True)
                break

        if not active_mask.any() and chunk_num == 0: continue

        batch_time = time.time() - outer_loop_start
        total_steps_in_batch = total_steps_this_batch.sum().item()
        total_steps_all_time += total_steps_in_batch
        sps = total_steps_in_batch / batch_time if batch_time > 0 else 0
        global_time = time.time() - start_time_all_time
        global_sps = total_steps_all_time / global_time if global_time > 0 else 0

        batch_max_steps, batch_max_idx = torch.max(total_steps_this_batch, 0)
        if batch_max_steps.item() > record_steps:
            record_steps = batch_max_steps.item()
            record_idx = batch_max_idx.item()
            record_gen = generation_num
            record_num = initial_x_words_this_gen[record_idx].cpu()
            torch.save(record_num, f"records/record_{record_steps}_steps_gen_{record_gen}.pt")
            logging.info(f"!!! NEW RECORD: {record_steps} steps from gen {record_gen} (idx {record_idx})")
        logging.info(f"    Longest chain so far: {record_steps} steps (gen {record_gen}, idx {record_idx})")

        logging.info(f"gen {generation_num}: batch terminated. "
                     f"Time: {batch_time:.2f}s, Steps: {total_steps_in_batch}, "
                     f"Speed: {sps/1e6:.2f} Msteps/s. Global avg: {global_sps/1e6:.2f} Msteps/s")

        # --- Update best originals pool ---
        if initial_is_original_mask_this_gen.any():
            current_gen_originals_words = initial_x_words_this_gen.view(torch.int64)[initial_is_original_mask_this_gen].view(DTYPE)
            current_gen_originals_fitness = total_steps_this_batch[initial_is_original_mask_this_gen]

            combined_words = torch.cat((best_originals_words, current_gen_originals_words))
            combined_fitness = torch.cat((best_originals_fitness, current_gen_originals_fitness))

            k = min(BEST_ORIGINALS_POOL_SIZE, combined_fitness.shape[0])
            top_fitness, top_indices = torch.topk(combined_fitness, k=k)

            best_originals_words = combined_words.view(torch.int64)[top_indices].view(DTYPE)
            best_originals_fitness = top_fitness

        # --- Evolutionary Step ---
        fitness = total_steps_this_batch
        num_children = BATCH_SIZE - num_elite - num_random

        next_gen_words = torch.empty_like(x_words)
        next_is_original_mask = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

        # Elitism: combined from current generation and best-originals pool
        elite_candidate_words = torch.cat((initial_x_words_this_gen, best_originals_words))
        elite_candidate_fitness = torch.cat((fitness, best_originals_fitness))
        is_original_from_current = initial_is_original_mask_this_gen
        is_original_from_pool = torch.ones(best_originals_words.shape[0], dtype=torch.bool, device=DEVICE)
        elite_candidate_is_original = torch.cat((is_original_from_current, is_original_from_pool))

        k_elite = min(num_elite, elite_candidate_fitness.shape[0])
        _, sorted_indices_for_elite = torch.topk(elite_candidate_fitness, k=k_elite)

        elite_words = elite_candidate_words.view(torch.int64)[sorted_indices_for_elite].view(DTYPE)
        elite_is_original = elite_candidate_is_original[sorted_indices_for_elite]

        next_gen_words[:k_elite] = elite_words
        next_is_original_mask[:k_elite] = elite_is_original

        if k_elite < num_elite:
            fill_count = num_elite - k_elite
            fill_words, _ = random_big_batched(time.time_ns(), n=fill_count)
            next_gen_words[k_elite:num_elite] = fill_words
            next_is_original_mask[k_elite:num_elite] = True

        # Random immigrants
        if num_random > 0:
            random_words, _ = random_big_batched(time.time_ns(), n=num_random)
            next_gen_words[num_elite : num_elite + num_random] = random_words
            next_is_original_mask[num_elite : num_elite + num_random] = True

        # Breeding
        if num_children > 0:
            p1 = tournament_selection(initial_x_words_this_gen, fitness, num_children, tournament_size=3)
            p2 = tournament_selection(initial_x_words_this_gen, fitness, num_children, tournament_size=3)
            # For mutation-only half, make parents identical to skip crossover.
            num_crossover = num_children // 2
            p2[num_crossover:] = p1[num_crossover:]
            import random
            bits_to_flip = random.choice([2,4,8,16])
            children_words = breed(p1, p2, bits_to_flip=bits_to_flip)
            next_gen_words[num_elite + num_random:] = children_words
            next_is_original_mask[num_elite + num_random:] = False

        perm = torch.randperm(BATCH_SIZE, device=DEVICE)
        x_words = next_gen_words.view(torch.int64)[perm].view(DTYPE)
        is_original_mask = next_is_original_mask[perm]
        find_hi_batched(x_words, x_hi)

        if not ran_tests:
            run_tests()
            run_extra_tests()
            ran_tests = True



if __name__ == "__main__":
    main()
