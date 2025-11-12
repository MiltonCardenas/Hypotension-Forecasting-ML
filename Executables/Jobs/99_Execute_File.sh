#!/bin/bash
#SBATCH --job-name=FSExecute_file
#SBATCH --output=/ihome/rparker/mdc147/PhD_Project/FeatureSelection.log
#SBATCH --error=/ihome/rparker/mdc147/PhD_Project/FeatureSelectionError.log
#SBATCH --time=13:00:00            #
#SBATCH --cpus-per-task=10         # 
#SBATCH --mem-per-cpu=10G          # 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=
#SBATCH --cluster=smp

source /ihome/rparker/mdc147/PhD_Project/Env_var/project_env/bin/activate
cd /ihome/rparker/mdc147/PhD_Project

#: Execution
python3 -u Executables/Models/17_2_FeatureSelection_LogLoss_AUPRC.py

