#!/bin/bash
# SLURM parameters ----------------------------------------
cluster="smp"                # Partition: htc, smp, gpu
TIME="8:00:00"               
CPUS=6                       # Cores
MEM=64G
JOB_NAME="basic_session"     # SLURM job name

# Environment /Project setup ------------------------------
PYTHON_ENV="/ihome/rparker/mdc147/PhD_Project/Env_var/project_env/bin/activate"
PROJECT_DIR="/ihome/rparker/mdc147/PhD_Project"
echo "Checking for an existing SLURM job named '$JOB_NAME'..."

JOBID=$(squeue --me --name="$JOB_NAME" --states=R -h -o "%A" | head -n 1)
if [ -z "$JOBID" ]; then
    echo "No active job found. Starting a new interactive job..."
    
    # SLURM session --------------------------------------
    exec srun --job-name="$JOB_NAME" \
              --time="$TIME" \
              --cluster="$cluster" \
              --cpus-per-task="$CPUS" \
              --mem="$MEM" \
              --pty /bin/bash --login -c "
                  module load java;
                  module load git/2.9.5;
                  source \"$PYTHON_ENV\";
                  cd \"$PROJECT_DIR\";
                  jupyter lab --no-browser --ip=0.0.0.0 --NotebookApp.token='';
                  exec /bin/bash
              "
else
    echo "Found existing job with ID: $JOBID"
    echo "Re-entering the session using srun..."
    exec srun --jobid="$JOBID" --pty /bin/bash --login -c "
        source \"$PYTHON_ENV\";
        cd \"$PROJECT_DIR\";
        exec /bin/bash
    "
fi