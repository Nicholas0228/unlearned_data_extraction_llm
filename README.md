# Breaking the Gold Standard: Extracting Forgotten Data under Exact Unlearning in Large Language Models

This repository provides the implementation of our algorithm for extracting unlearned data from large language models (LLMs) using guidance-based methods. The code is built primarily upon the [TOFU repository](https://github.com/locuslab/tofu) and includes data from [MUSE](https://muse-bench.github.io).

The core implementation can be found in `MUSE/evaluate_util.py`, particularly the `contrasting_generation` function.

## Requirements

We follow most of the dependencies used in TOFU. To set up the environment:

```bash
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Usage

### Step 1: Prepare models before and after unlearning

```bash
cd MUSE
bash finetune_phi_all_iter_v2.sh
```

This step takes approximately 12 hours on 2×A100 GPUs.

### Step 2: Perform extraction

```bash
bash eval_idea_10_v2.sh
```

This will measure the memorization of the forgetting set and save the results to the corresponding checkpoint directory.

### Step 3: Evaluate the results

```bash
python read_final_res.py
```

This script outputs a comparison between the pre- and post-unlearning models, along with the performance of our extraction method.

