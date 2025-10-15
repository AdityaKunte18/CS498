###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    work = [c.clone() for c in chunks]
    recv_buf = torch.empty_like(tmp)

    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world

        tmp.copy_(work[send_idx])
        send_req = dist.isend(tmp, dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        recv_req.wait()
        work[recv_idx] += recv_buf
        send_req.wait()

    cur = (rank + 1) % world
    chunks[cur].copy_(work[cur])
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    recv_buf = torch.empty_like(tmp)

    cur = current                      
    tmp.copy_(chunks[cur])           

    for _ in range(world - 1):
        send_req = dist.isend(tmp, dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        recv_req.wait()
        recv_idx = (cur - 1) % world
        chunks[recv_idx].copy_(recv_buf)

        send_req.wait()
        cur = recv_idx                 
        tmp.copy_(chunks[cur])

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    
    
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padded_len = chunk * world
    pad = padded_len - n
    if pad > 0:
        padded_flat = torch.cat([flat,torch.zeros(pad)])
    else:
        padded_flat = flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    tmp = torch.empty_like(chunks[0])  # send buffer
    reduce_scatter(chunks, tmp, world, rank, left, right)
    current = (rank + 1) % world 
    all_gather(chunks, tmp, current, world, rank, left, right)
    flat = torch.cat(chunks, dim=0)
    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return