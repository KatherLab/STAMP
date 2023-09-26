#!/bin/bash
#SBATCH --job-name=wsi-pipeline_TCGA-BRCA-E2                     # create a short name for your job
#SBATCH --account=p_scads_pathology                 # account the computing resources are assigned to
# partition on HPC cluster, batch system minimizes number of nodes
#SBATCH --time=21:00:00                             # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                                   # node count
#SBATCH --ntasks=1                                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=32                          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=200G
#SBATCH --mail-type=end                             # send email when job ends
#SBATCH --mail-user=omar.el_nahhas@tu-dresden.de      # mail adress
#SBATCH --output=results.txt


srun singularity run --nv -B /scratch/ws/0/omel987e-wsi-pipeline:/opt/data /scratch/ws/0/omel987e-wsi-pipeline/hpc_cenv.sif $1
