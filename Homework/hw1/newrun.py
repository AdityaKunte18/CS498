#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cpu_hybrid_pp.py

This is the main experiment script for training Qwen-0.6B using a custom 
Hybrid Pipeline Parallelism framework built on top of PiPPy.

Key Features:
- CPU-only execution with 3 ranks
- Manual microbatch scheduling via custom action list
- Used for testing correctness and loss behavior of the hybrid PP runtime
"""
import math
import hashlib, datetime as dt
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time

from q2 import PS_grads_
from q3 import ring_allreduce_

###a example of LLM templete for reference###
###you don't have to actually use it###
class LLMTemplete(nn.Module):                      
    def __init__(self, model, end):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:end])     
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb               

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        hidden = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = torch.triu(
            torch.Qwenmodel((seqlen, seqlen), float('-inf'), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_outputs[0]
        hidden = self.norm(hidden)


        return self.lm_head(hidden)


###parser definition###
###you don't need to use it, the batch_size is fixed to 8
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 8)),
                    help="Batch size of each rank/node, not Global batch size")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of training epochs")   # <-- NEW
parser.add_argument("--question", type=int, default=3,
                    help="specify parameter server (2) or all reduce (3)")   # <-- NEW
args = parser.parse_args()


###dataset generation###
def globaldataloader_generation(tokenizer, batch_size, len_max = 256): # length of sample is fixed to 256
    ###generate a simple training task###
    ###we fetch wikitext dataset from huggingface###
    rawDataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def tok_fn(ex): 
        return tokenizer(ex["text"], return_attention_mask=False)
    def grp_fn(ex):
        concat = sum(ex["input_ids"], [])
        tot = len(concat) // len_max * len_max
        ids = [concat[i:i+len_max] for i in range(0, tot, len_max)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}
    
    ds = (rawDataset.map(tok_fn, batched=True, remove_columns=["text"])
              .map(grp_fn, batched=True))
    ds = ds.select(range(3*24)) #totally 72 samples
    ds.set_format("torch", columns=["input_ids", "labels"])
    
    return ds


# -------------------------------
### Helpers for unit test/grading(do not modify)
# -------------------------------
### these codes are for unit test. You don't have to understand
def flat_params(model: nn.Module):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def sha256_chunks_u64(t: torch.Tensor):
    """Return 4x uint64 from SHA-256 of tensor bytes (small, fixed-size gather)."""
    arr = t.detach().cpu().numpy().tobytes()
    h = hashlib.sha256(arr).digest()  # 32 bytes
    parts = [int.from_bytes(h[i:i+8], "little", signed=False) for i in range(0, 32, 8)]
    return torch.tensor(parts, dtype=torch.uint64)

def assert_models_identical(model: nn.Module, world: int, rank: int):
    vec = flat_params(model)
    sig_local = sha256_chunks_u64(vec).to(torch.int64)       
    gather_list = [torch.empty_like(sig_local) for _ in range(world)]
    dist.all_gather(gather_list, sig_local)                 #Noticed: the pytorch built-in all_gather is only used for unit test check(not allowed to use in your all-reduce implementaion)
    # Rank-0 checks all signatures equal
    if rank == 0:
        for i in range(1,dist.get_world_size()):
            ok = torch.equal(gather_list[0], gather_list[i])
            if not ok:
                raise AssertionError(f"Model checksums of rank0 differ across ranks:\n{i}")
        print("identical model test passed!:white_check_mark:")


###rest helper utils###
def shard_slice(n: int, rank: int, world: int):
    per = math.ceil(n / world)
    s, e = rank * per, min((rank + 1) * per, n)
    return s, e

###provided function: we do all-reduce sync for each layer###
###important: your ring_allreduce_ is called here###
###we do the all-reduce warpper for you### 
def allreduce_grads_ring_(model: nn.Module, world_size=None, rankid=None, opt=None):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads: return
    flat = _flatten_dense_tensors(grads)
    ring_allreduce_(flat, world_size = world_size, rankid = rankid)
    synced = _unflatten_dense_tensors(flat, grads)
    for g, s in zip(grads, synced):
        g.copy_(s)
    opt.step()

def _allclose_across_ranks(t: torch.Tensor, world: int, atol=1e-6, rtol=1e-5):
    """Assert all ranks hold the same tensor `t` (testing-only; uses all_gather)."""
    v = t.detach().cpu()
    vs = [torch.empty_like(v) for _ in range(world)]
    dist.all_gather(vs, v)
    ref = vs[0]
    for i in range(1, world):
        if not torch.allclose(ref, vs[i], atol=atol, rtol=rtol):
            raise AssertionError(f"Tensors differ across ranks at idx {i}")

def _rankvec(n: int, rank: int, dtype=torch.float32, device="cpu"):
    """Deterministic per-rank vector to test averaging (different content per rank)."""
    # linear ramp + rank offset makes per-rank data distinct but easy to verify
    base = torch.arange(n, dtype=dtype, device=device)
    return base + (rank + 1) * 0.1

def _test_ring_allreduce_synthetic(world: int, rank: int):
    """Quick ring all-reduce tests that stress padding, odd sizes, and contiguity."""
    from q3 import ring_allreduce_

    lengths = [
        1, 2, 3,                    # tiny
        31, 32, 33,                 # around power-of-two
        127, 128, 129,              # larger boundaries
        257,                        # prime-ish
        997                         # bigger odd
    ]
    for n in lengths:
        x = _rankvec(n, rank)                   # distinct per rank
        flat = x.clone()

        # Expected average = mean across ranks of the originals
        # gather all to rank 0 (testing-use only)
        xs = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(xs, flat)
        if rank == 0:
            expected = torch.stack(xs, dim=0).mean(dim=0)
        else:
            expected = torch.empty_like(flat)   # will be filled

        # run ring allreduce IN-PLACE on `flat`
        ring_allreduce_(flat, world_size=world, rankid=rank)

        # broadcast expected to all ranks (testing)
        dist.broadcast(expected, src=0)

        # check numeric correctness and shape restored
        assert flat.shape == (n,), f"shape changed after ring_allreduce for n={n}"
        if not torch.allclose(flat, expected, atol=1e-6, rtol=1e-5):
            raise AssertionError(f"ring_allreduce mismatch for n={n}")

        # non-contiguous input path: transpose a (n,1) to (1,n) and view back
        # (forces a non-contiguous intermediate)
        y2d = x.view(n, 1).T   # non-contiguous
        y = y2d.contiguous().view(-1).clone()   # simulate caller passing a flat view again
        ring_allreduce_(y, world_size=world, rankid=rank)
        # y should still equal expected
        if not torch.allclose(y, expected, atol=1e-6, rtol=1e-5):
            raise AssertionError(f"ring_allreduce mismatch (non-contig path) for n={n}")

    if rank == 0:
        print("[ring] synthetic tests passed ✓")

def _test_ps_step_once_only(world: int, rank: int):
    """Check PS semantics: grads averaged, optimizer step only on server,
    but final params equal everywhere and match an explicit average update.
    """
    from q2 import PS_grads_
    device = torch.device("cpu")

    # tiny model to keep it fast/deterministic
    m = torch.nn.Linear(7, 3, bias=True).to(device)
    torch.manual_seed(1234)
    # snapshot initial params to compute explicit expected update
    with torch.no_grad():
        init = torch.cat([p.detach().view(-1).clone() for p in m.parameters()])

    # fabricate deterministic per-rank grads (no forward/backward needed)
    grads = []
    for p in m.parameters():
        g = torch.full_like(p.data, fill_value=(rank + 1) * 0.5)  # rank-scaled constant grad
        grads.append(g)
    # write into .grad
    for p, g in zip(m.parameters(), grads):
        p.grad = g.clone()

    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.0)

    # Run PS (server averages grads, steps, then sends updated params)
    PS_grads_(m, world_size=world, rankid=rank, opt=opt)

    # Compute the explicit expected params on rank 0 and broadcast
    # Expected grad = mean over ranks of the fabricated grads:
    # mean value per tensor = 0.5 * mean(rank+1) = 0.5 * ((1+world)/2)
    mean_scale = 0.5 * ( (1 + world) / 2.0 )
    with torch.no_grad():
        exp_params = []
        for p_init, p in zip(_unflatten_dense_tensors(init, list(m.parameters())),
                             list(m.parameters())):
            # SGD: p_new = p_init - lr * grad_avg
            exp_params.append((p_init - 0.1 * torch.full_like(p_init, mean_scale)).view(-1))
        exp_params = torch.cat(exp_params)

    # gather actual params
    actual = torch.cat([p.detach().view(-1) for p in m.parameters()]).cpu()
    # place expected on rank 0 and broadcast
    if rank != 0:
        exp_buf = torch.empty_like(actual)
    else:
        exp_buf = exp_params.detach().cpu().clone()
    dist.broadcast(exp_buf, src=0)

    # assert equality everywhere
    if not torch.allclose(actual, exp_buf, atol=1e-6, rtol=1e-6):
        raise AssertionError("PS updated parameters do not match expected average-SGD update.")

    # and assert all ranks are identical (paranoid)
    _allclose_across_ranks(actual, world)
    if rank == 0:
        print("[ps] step-once-only + averaging test passed ✓")

###main work###
def main():

    print("waiting distributed init setup...")
    ###set up global distributed config###
    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print("running preflight robustness tests...")
    _test_ring_allreduce_synthetic(world, rank)
    _test_ps_step_once_only(world, rank)
    dist.barrier()
    if rank == 0:
        print("preflight tests passed, begin model loading...")

    print("dist group setup finished, begin model loading...")
    ###get pretrained qwen3-0.6b model from huggingface###
    device = torch.device("cpu")     
    name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    Qwenmodel = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    ###generate local dataset loader###
    len_max = 256
    ds = globaldataloader_generation(tokenizer, args.batch_size, len_max = len_max)

    start, end = shard_slice(len(ds), rank, world)
    local_ds = torch.utils.data.Subset(ds, range(start, end))
    loader = DataLoader(local_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)


    
    print(f"per device batch size is {args.batch_size}, global batch size is {args.batch_size*world}.")
    print(f"synthesized dataset with per sample's len = 256 finished successfully. ")
    print(f"each local dataset contains {len(loader)} samples.")
    print(f"node {rank} is assigned by sample's no from {start} to {end}.")

    ###get rid of randomness: do not modify this line###
    torch.manual_seed(42)
    
    Qwenmodel.to(device)
    Qwenmodel.train()

    opt = optim.Adam(Qwenmodel.parameters(), lr=1e-4)    #set optimizor
    ce_sum = nn.CrossEntropyLoss(reduction="sum")


    dist.barrier()

    time_stamp = []
    for epoch in range(args.epochs):
        run_loss_sum_local = 0.0
        run_tok_local = 0.0
        start_time = time.time()
        step = 0

        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            opt.zero_grad(set_to_none=True)

            out = Qwenmodel(input_ids=input_ids, use_cache=False, output_hidden_states=False)
            logits = out.logits  # [B, T, V]
            B, T, V = logits.shape
            # shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            local_token_amount = B * (T - 1)

            loss_sum = ce_sum(shift_logits.view(-1, V), shift_labels.view(-1))  # scalar SUM over tokens
            loss = loss_sum / local_token_amount
            loss.backward()

            if args.question == 2:
                ###do sync for each layers' parameter...###
                ###important: your code will be called here..
                pre_vec = flat_params(Qwenmodel).clone()
                print(f"epoch {epoch} step {step} begin gradient parameter server sync...")
                print("Using parameter server to sync...")
                PS_grads_(Qwenmodel, world_size = world, rankid =rank, opt=opt)
                print("parameter server sync finished.")
                # ---- ASSERT #2: Parameters actually changed (not a no-op) ----
                post_vec = flat_params(Qwenmodel)
                delta_norm = (post_vec - pre_vec).norm().item()
                if rank == 0:
                    # Non-finite or zero update usually means grads weren’t averaged/applied correctly
                    assert math.isfinite(delta_norm), "[PS check] non-finite parameter delta after PS step"
                    assert delta_norm > 0, "[PS check] zero parameter delta; PS step may not be applying updates"
            elif args.question == 3:
                ###do sync for each layers' parameter...###
                print(f"epoch {epoch} step {step} begin all reduce sync...")
                print("Using all reduce to sync...")
                allreduce_grads_ring_(Qwenmodel, world_size = world, rankid =rank, opt=opt)
                print("all reduce sync finished.")


            ### Logging (average loss per token over time)###
            run_loss_sum_local += loss_sum.item()
            run_tok_local += local_token_amount
        
        print(f"rank {rank} epoch {epoch} aggregated loss/token={run_loss_sum_local/run_tok_local:.4f}")
        dist.barrier()

        if rank == 0:
            total_time = time.time() - start_time
            time_stamp.append(total_time)
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average step speed: {step / total_time:.2f} steps/s")



        ###unit test###
        ###this test checks that if final models across nodes are identical###
    assert_models_identical(Qwenmodel, world = world, rank = rank)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()