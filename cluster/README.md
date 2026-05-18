# Cluster jobs (SLURM)

Quick reference for running experiments on a SLURM cluster. All scripts are launched from the **repo root**.

## First-time setup

On the login node, after cloning the repo:

```bash
cd <path>/ssl-cp-research

# Find out what partitions / accounts you have:
sinfo -o "%P %a %l %D %N"            # partitions, max walltime, nodes
sacctmgr show user $USER -s          # account / qos visible to you

# Pick ONE python setup path, then edit cluster/extract_cifar100.sbatch
# accordingly (uncomment Option A / B / C in the env block):

# (A) Conda — once only:
bash cluster/setup_env.sh

# (B) Modules + venv — adjust module names to your cluster:
#   module load python/3.10 cuda/11.8
#   python -m venv ~/venvs/ssl-cp
#   source ~/venvs/ssl-cp/bin/activate
#   pip install -r requirements.txt

mkdir -p cluster/logs output data
```

## Submit a job

```bash
sbatch cluster/extract_cifar100.sbatch
squeue -u $USER                      # watch your queue
tail -f cluster/logs/extract_cifar100_*.out
```

## After it finishes

```bash
ls -lh output/embeddings_cifar100.pt
# Embeddings are gitignored — keep them on the cluster's scratch/storage.
```
