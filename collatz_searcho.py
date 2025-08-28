#!/usr/bin/env python3
import time, torch, sys

"""
Simple local base-6 cellular-automaton Collatz on GPU:
- Digits 0..5, BLANK=6
- Store numbers MSB->LSB left->right with trailing BLANKs on the right
- One-step update is purely local: new[i] = F(left=a[i-1], self=b[i]) as per Mathematica rule
- Always ensure right-side headroom by appending BLANK columns when tail gets close
- Loop forever: sample batch, save lastbatch.pt, evolve until each row is exactly a single '1'
- Track per-row step counts; on a new record, validate by CPU Collatz and save bestbatch.pt
- Log progress every PROGRESS_EVERY steps, including minmag/maxmag (row lengths in digits)
"""

DEVICE = 'cuda'
DT_DIG = torch.int16
DT_STEPS = torch.int32
BATCH = 1024 * 8
NUM_DIGITS = 1024 * 1
BLANK = 6
PROGRESS_EVERY = 5000
MARGIN = 64

def log(msg):
    "Prints a timestamped message."
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def ensure_right_margin(x, margin=MARGIN):
    "Appends BLANK columns on the right if any non-BLANK is within the last margin."
    if x.shape[1] < margin:  # trivial small case
        pad = torch.full((x.shape[0], margin), BLANK, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)
    tail_busy = x[:, -margin:].ne(BLANK).any()
    if tail_busy:
        pad = torch.full((x.shape[0], margin), BLANK, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
    return x

def sample_batch(batch=BATCH, digits=NUM_DIGITS, device=DEVICE):
    "Samples random base-6 numbers (MSB-left) with trailing BLANKs to the right."
    B = batch; L = digits
    x = torch.full((B, L), BLANK, dtype=DT_DIG, device=device)
    lens = torch.randint(1, max(2, L // 4), (B,), device=device)
    # fill 0..len-1 with random digits, then force MSB nonzero
    vals = torch.randint(0, 6, (B, L), dtype=DT_DIG, device=device)
    idx = torch.arange(L, device=device).unsqueeze(0)
    mask = idx < lens.unsqueeze(1)
    x[mask] = vals[mask]
    r = torch.arange(B, device=device)
    x[r, 0] = torch.randint(1, 6, (B,), dtype=DT_DIG, device=device)
    # ensure headroom on right
    x = ensure_right_margin(x)
    return x, lens

def step_local(x):
    "One CA step via local rule using left neighbor a and self b (MSB-left)."
    B, L = x.shape
    a = torch.cat([torch.full((B,1), BLANK, dtype=x.dtype, device=x.device), x[:, :-1]], dim=1)
    b = x
    # core update when b != BLANK
    v = 3 * (a.remainder(2)) + torch.div(b, 2, rounding_mode='floor')
    v = torch.where((a.eq(BLANK)) & (v.eq(0)), torch.full_like(v, BLANK), v)
    # when b == BLANK: if a even -> BLANK else 4
    newb = torch.where(b.eq(BLANK), torch.where(a.remainder(2).eq(0), torch.full_like(b, BLANK), torch.full_like(b, 4)), v)
    return newb.to(DT_DIG)

def count_nonblank(x):
    "Counts non-BLANK per row."
    return x.ne(BLANK).sum(dim=1)

def row_lengths(x):
    "Returns number of non-BLANK digits per row."
    return count_nonblank(x)

def digits_to_int_row_msb(row):
    "Converts one MSB-left BLANK-padded base-6 row to int."
    row = row.tolist()
    n = 0
    for d in row:
        if d == BLANK: break
        n = n * 6 + int(d)
    return n

def collatz_steps_cpu(n):
    "Counts CPU Collatz steps to reach 1."
    c = 0
    while n > 1:
        n = n // 2 if (n & 1) == 0 else 3 * n + 1
        c += 1
    return c

def run_forever():
    "Main loop: resample, save, simulate with progress, track and validate records."
    best_alltime_chainlen = -1
    best_alltime_seed_and_index = [0, 0]
    batch_num = 0
    while True:
        batch_num += 1
        seed = time.time_ns()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x0, _ = sample_batch()
        torch.save({'init_digits_msb': x0.detach().to('cpu'), 'seed': seed}, 'lastbatch.pt')
        x = x0.clone()
        steps = torch.zeros(x.shape[0], dtype=DT_STEPS, device=x.device)
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        t = 0
        log(f'batch {batch_num}: start; seed={seed} rows={x.shape[0]} digits={x.shape[1]}')
        while True:
            if not torch.any(~done): break
            x_next = step_local(x)
            x = torch.where(done.unsqueeze(1), x, x_next)
            x = ensure_right_margin(x)
            t += 1
            steps = steps + (~done).to(DT_STEPS)
            nb = count_nonblank(x)
            ones = x.eq(1).sum(dim=1)
            newly_done = (nb == 1) & (ones == 1)
            done = done | newly_done
            if t > 0 and t % PROGRESS_EVERY == 0:
                active_rows = x[~done]
                lens = row_lengths(active_rows) if active_rows.shape[0] > 0 else torch.tensor([0], device=x.device)
                remaining = (~done).sum().item()
                finished = done.sum().item()
                best_now = steps[done].max().item() if finished > 0 else 0
                log(f'batch {batch_num}: t={t} finished={finished}/{x.shape[0]} remaining={remaining} best_finished_steps={best_now} minmag={int(lens.min().item())} maxmag={int(lens.max().item())}')
        best_batch_chainlen, best_batch_idx = steps.max(dim=0)
        best_batch_chainlen = int(best_batch_chainlen.item())
        best_batch_idx = int(best_batch_idx.item())
        best_batch_seed_and_index = [seed, best_batch_idx]
        log(f"Batch {batch_num} best: {best_batch_chainlen} steps (seed,index)={best_batch_seed_and_index}")
        if best_batch_chainlen > best_alltime_chainlen:
            log(f"New all-time record: {best_batch_chainlen} steps!")
            best_alltime_chainlen = best_batch_chainlen
            best_alltime_seed_and_index = best_batch_seed_and_index
            row_init = x0[best_batch_idx].detach().to('cpu')
            n = digits_to_int_row_msb(row_init)
            cpu_steps = collatz_steps_cpu(n)
            log(f'Validating: gpu_steps={best_batch_chainlen} cpu_steps={cpu_steps} n={n}')
            if best_batch_chainlen != cpu_steps:
                log(f"!!! VALIDATION FAILED: GPU={best_batch_chainlen} CPU={cpu_steps} !!!")
                sys.exit(1)
            torch.save({'init_digits_msb': row_init, 'seed_and_index': best_alltime_seed_and_index, 'gpu_steps': best_batch_chainlen, 'cpu_steps': cpu_steps, 'n': n}, 'bestbatch.pt')
        log(f"All-time best: {best_alltime_chainlen} steps (seed,index)={best_alltime_seed_and_index}")

if __name__ == '__main__':
    torch.cuda.init()
    run_forever()
