# Introduction for annuna server

## Overview of annuna HPC (High Performance Computing )
- Quite few GPU(22), will be updated soon.
- Software
    - Operating System:Ubuntu 20.04
    - Scheduler:SLURM
    - Compilters: GCC, Intel, ...
    - Software: R, Octave, Matlab, julia, python, OpenFOAM, AlphaFold, ..
- Don’t run code in VScode, just for editing files or using sinteractive karnel. 

## Slurm basic command

- `sinfo` check available node.
- `sbatch sbatch_training.sh` run sbatch command
- `squeue -a` check current runing program
- `scancel <id>` cancel task
- `sinfo -N -l` available data
- `sacct -j <id>` check task condition

## Jobs by Sinteractive
- Sequential or parallel
- If parallel: multi-threaded or multi-process?
- Resource requests:
    - Number of CPUS
    - Amount of RAM
    - Expected computing time (if overtime, it will be killled.)
- sinteractive
    - Wrapper on srun
    - Provides an immediate interactive shell on the node
    - `sinteractive -c <cpus> --mem <MB>`
    - `sinteractive -c 1 --mem 2000M`
    - —time=day-hour:minutes : default is hour
    
    ```bash
    sinteractive --mem 100M --time=0-0:10 -c 1
    ```

### SSH tunnel to a certain node
* Submit a slurm job, or start an interactive job on a node.
`sinteractive`
* or for a Nvidia A100 GPU:
`sinteractive -p gpu --gres=gpu:1 --constraint='nvidia&A100'`
* Check on which node this job has landed
`squeue -u $USER -l`
* after activate interractive node, activate python environment
` source chm_env/bin/activate`
* Add this to your machines ssh config in ~/.ssh/config
```bash
Host node* gpu*
    Hostname       %h.internal.anunna.wur.nl
    ProxyJump      login.anunna.wur.nl
    User           ishik001

Host *.anunna.wur.nl
    User           ishik001 
```
You can now use your local terminal connecting to e.g. node201 directly, by:
`ssh node201`
In VSCode :
in the VSCode GUI -> Open a remote Window -> Connect to Host...,type in the nodename. After a few seconds, VSCode is connected to the node.
- Please be aware that once the slurm job finishes, the connection will be lost and your running processes will be killed. Besides that, you are limited to the amount of RAM that you requested.


## Jobs SLURM
- run code
    - `sbatch </Path/To/Script.sh>`
    - SLURM assigns an ID to the job ($JOBID)
- `squeue -u $USER`
- `squeue -a` can see what jobs are running at this moment50.
- `scancel -u $USER`
- maximum time is set to 3 weeks
- Monitoring jobs
 - `sacct -X -u $USER --starttime 2024-01-01 --endtime now`
 - `scontrol show job 54463146`

- example of sbatch
```bash
#!/bin/bash

#SBATCH --job-name=finetune_tsc
#SBATCH --output=output/finetune_log/output_%j.txt
#SBATCH --time=0-5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --error=output/finetune_log/output_error_%j.txt
##SBATCH --mail-type=ALL
##SBATCH --mail-user=takayuki.ishikawa@wur.nl

# Create output directory if it doesn't exist
mkdir -p output/finetune_log

# Run the finetuning script with necessary arguments
python finetune_presto_with_pretrained_model.py \
    --seed_list 5 6 7 8 9\
    --re_pretrain_datasize_list 0 53062 103062 203062 1003062 \
    --num_layers 3 \
    --nodes 128 \
    --is_nodes_half \
    --dropout 0.2 \
    --lr 0.00057 \
    --weight_decay 0.00746 \
    --patience 200 \
    --downstream_eval_per_best_val_loss 300 \
    --finetune_batch_size 64 \
    --finetune_epochs 200 \
    --train_ratio 0.50 \
    --stratified_split \
    --min_learning_rate 0.0001 \
    --wandb \
    --wandb_org "wildflowers315-wageningen-university-research" \
    --create_slide \
    --s1s2only \
    --random_mask \
    --label_ts "Group_F" \
    --wandb_project "finetune_pretrained_Group_F_pure_HO" \
    --is_pure_plot

# Optional arguments (commented out):
    # --s1s2only                 # Use only S1 and S2 bands (exclude ERA5 and SRTM)
    # --freeze_encoder           # Freeze encoder weights during finetuning

# Notes:
# - Adjust time limit (--time) based on expected runtime
# - Modify memory (--mem-per-cpu) if needed
# - Update wandb project name for different experiments
```
