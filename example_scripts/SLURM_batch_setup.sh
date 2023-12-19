#!/bin/bash
#SBATCH --job-name=STAMP_run_0                      # create a short name for your job
#SBATCH --account=your_partition                    # account the computing resources are assigned to
# partition on HPC cluster, batch system minimizes number of nodes
#SBATCH --time=21:00:00                             # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks=1                                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=130G
#SBATCH --mail-type=end                             # send email when job ends
#SBATCH --mail-user=you@email.com                   # mail adress
#SBATCH --output=results_run_0.txt

srun singularity run --nv -B /:/ /path/to/STAMP_container.sif $1
#run outside this script as, for example:
#sbatch SLURM_preprocessing.sh "stamp preprocess"
