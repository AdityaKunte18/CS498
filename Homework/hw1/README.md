# Question 2 and 3 Instructions

For **q2** and **q3**, you will implement **Parameter Server (PS)** and **All-Reduce** synchronization for the **Qwen3** model’s training under a **3-node Data Parallel (DP)** setup in `q2.py` and `q3.py`.
Most of the training workflow is already implemented in `run.py`. Once you finish a sync method, run `run.py` to verify basic correctness.

> **Start early!** Each training run takes several minutes. When multiple students share a cluster, queueing will slow things down.

---

## Experiment Setup Instructions - Cloudlab

We created multiple clusters and assigned each student to **one** cluster. We have assigned students to clusters based on their last name. **Please run your experiments only on your assigned cluster** to avoid overloading others. If students run into any issues due to the clusters getting overloaded, please email us, and we will assign you to new cluster that has less congestion.

Cluster Assignments:
* HW1-Cluster-1: Alaybeyi-Hong
* HW1-Cluster-2: Hu-Lu
* HW1-Cluster-3: Luo-Sankaran
* HW1-Cluster-4: Sen-Wu
* HW1-Cluster-5: Xie-Zhuang

1. **Familiarize yourself with CloudLab**

   * Go to **`Experiment → Topology View`** to see the network layout (triangle topology).
   * Click a node, then **`Shell`** to you can open an in-browser SSH session.
   * To SSH from your own terminal, upload your key via the top-right menu **Manage SSH Keys**. Then use **`Experiment → List View`** to copy the SSH command for a node.

2. **Python environment**

   * We recommend using **conda** with **Python ≥ 3.10.12** to avoid dependency issues.
   * Install required packages:

     ```bash
     sudo apt-get update
     sudo apt-get install -y python3-pip
     pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     pip install transformers datasets
     echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
     source ~/.bashrc
     ```

3. **Know PyTorch Gloo**

   * The project uses PyTorch’s **`gloo`** distributed backend.
   * To ensure nodes communicate over the experiment LAN, set the network interface for Gloo on **each** node:

      1. Run `ifconfig` and identify the interface connected to the experiment network (typically named like `en*`, e.g., `enp4s0f1`, with a `10.x.x.x` address. There may be several candidates, you can choose one).
      2. For each node, export the interface name and master address, which you can choose, but for simplicity, we recommend choosing node-0's address.
        ```bash
        export GLOO_SOCKET_IFNAME=enp4s0f1
        export MASTER_ADDR=10.x.x.x
        ```
      * **Important:** If this variable is not set correctly, the distributed job maybe hang forever, and you will need to close the terminal session and reopen. 
      * **Note** We recommend to set the `MASTER_ADDR` stays the same across nodes, and the `GLOO_SOCKET_IFNAME` is the same among worker nodes, and is different from the `GLOO_SOCKET_IFNAME` of the master node. However, during the ping test, if you encounter `Destination Host Unreachable` error on any of the worker nodes, just switch the network interface to the same one used in the master node.
      * Example: Let's use node-0 as the master-node. In this example, `ifconfig` on node-0 gives us a node interface `enp94s0f0` with an inet of `10.10.2.2`, and node interface `enp94s0f1` with an inet of `10.10.1.2`. `ifconfig` on node-1 gives us a node interface `enp94s0f0` with an inet of `10.10.3.2`, and node interface `enp94s0f1` with an inet of `10.10.1.1`. `ifconfig` on node-2 gives us a node interface `enp94s0f0` with an inet of `10.10.3.1`, and node interface `enp94s0f1` with an inet of `10.10.2.1`. We set the master address in all 3 nodes with `export MASTER_ADDR=10.10.2.2`. We set the node-0's gloo socket with `export GLOO_SOCKET_IFNAME=enp94s0f0`, and node-1 and node-2's gloo sockets with `export GLOO_SOCKET_IFNAME=enp94s0f1`.
     3. To check if this was done properly, you can run `ping -I $GLOO_SOCKET_IFNAME $MASTER_ADDR` on all nodes, and it should run without issue.
      * If you encounter `Destination Host Unreachable` error on any of the worker nodes, just switch the network interface to the same one used in the master node.

  ## How to run training:

1. **overall of commands:**
   You can assign rank indices to nodes yourself (by default, `node-0 → rank 0`, `node-1 → rank 1`, `node-2 → rank 2`, but you may swap if you prefer).

   The command **template** is:

   ```bash
   # For Rank 0:
   ./glooHelper.sh -n 3 -P 1 -r 0 -e 1 -q <your question number (2 or 3)> -s ./run.py

   # For Rank 1:
   ./glooHelper.sh -n 3 -P 1 -r 1 -e 1 -q <your question number (2 or 3)> -s ./run.py

   # For Rank 2:
   ./glooHelper.sh -n 3 -P 1 -r 2 -e 1 -q <your question number (2 or 3)> -s ./run.py
   ```

2. **command step by step:**

   We provide a helper script `glooHelper.sh` to simplify launching. The script derives a socket port for `gloo` from the contents of `netid.txt` (place your NetID in that file) to avoid port contention across students.

   With the helper script, fill in these options:

   * `-n`: number of nodes/ranks. **Use 3**.
   * `-P`: number of processes per node (PyTorch’s `--nproc-per-node`). **Use 1**.
   * `-r`: the **rank index** of this node (`0–2`). The node with **rank 0** acts as the master. For `gloo`, the initial distributed setup and subsequent coordination are handled by the master node.
   * `-e`: the number of epochs, we set it to `1` by default
   * `-q`: the question number (whether you are working on `2` or `3`)


3. After filling all options, run the command on **each node**. All options should be identical across nodes **except** for the rank index (`-r`).

   Before implementing q2 and q3 sync methods, you can run the training framework first to verify your configuration, by setting the number of epochs to 0. If the job forms the distributed group successfully and then loads the model and dataset without errors, you’re good to proceed.

## Tips for your work:

All of your coding work is in `q2.py` and `q3.py`. However, understanding the overall training flow will help. In `run.py`, we construct a synthetic dataset for Qwen3 fine-tuning training. Each sample has a fixed length of **256**. The total dataset size is **3 × 24** (so each node gets **24** samples). We fix the batch size to **8**, so each epoch has **24 / 8 = 3** steps. Training runs for three epochs: **epoch 0** uses the PS sync method, and **epochs 1–2** use All-Reduce. After each epoch, the aggregated loss per token is printed for debugging.

1. **Read the comments.** We’ve added essential comments to guide you through the training flow. Also review the slides on **All-Reduce** and **Parameter Server**. In `run.py`, focus on the provided `allreduce_grads_ring_` wrapper and the code starting at **line 216**—that’s where the sync method is applied.

2. **Know `nn.Module` basics.** You should be comfortable extracting parameters and grads:

   ```python
   params = [p for p in model.parameters() if p.grad is not None]
   grads  = [p.grad for p in params]
   ```

   See the PyTorch documentation for details on modules, parameters... https://docs.pytorch.org/docs/stable/pytorch-api.html

3. **Available tools.** You **may not** use PyTorch’s direct collectives (e.g., `dist.all_reduce`) for this assignment. With `import torch.distributed as dist` available, implement your own sync using **non-blocking P2P**:

   ```python
   # Rank 0:
   send_buf = torch.tensor([1, 2, 3])
   s = dist.isend(send_buf, dst=2)   # send to rank 2
   s.wait()

   # Rank 2:
   recv_buf = torch.empty(3, dtype=torch.long)
   r = dist.irecv(recv_buf, src=0)   # receive from rank 0
   r.wait()

   # Always wait before reusing/read/writing the buffers
   # after wait, rank 2's recv_buf == tensor([1, 2, 3])
   ```

   Also, for easier chunking/merging, use:

   * `torch._utils._flatten_dense_tensors(list_of_tensors)`
   * `torch._utils._unflatten_dense_tensors(flat_tensor, like_list)`
     These help turn a list of high-dim tensors into a single 1-D buffer (and back).

4. **Unit test in `run.py`.** A basic test at the end of `run.py` checks whether the final model parameters are identical across the three nodes.

   * To test PS only, set the -q argument to `2`.
   * To test All-Reduce only, set the -q argument to `3`.
     You may modify `run.py` as needed for your own testing; you do **not** need to submit `run.py`. All graded code changes are in `q2.py` and `q3.py`
   * In `q3.py`, you may modify function signature (input structure) of `reduce_scatter` and `all_gather` if needed for you to better implement `ring_allreduce_`.
