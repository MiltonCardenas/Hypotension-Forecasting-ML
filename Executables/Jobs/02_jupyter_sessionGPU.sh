#!/bin/bash
#: SLURM parameters ------------------------------
PARTITION="gtx1080"          # GPU partition: a100, a100_nvlink
TIME="4:00:00"               
CPUS=2                       
MEM=40G
JOB_NAME="basic_session2"   

#: Environment / Project setup -------------------
PYTHON_ENV="/ihome/rparker/mdc147/PhD_Project/Env_var/project_env/bin/activate"
PROJECT_DIR="/ihome/rparker/mdc147/PhD_Project"
echo "Checking for an existing SLURM job named '$JOB_NAME'..."
JOBID=$(squeue --me --name="$JOB_NAME" --states=R -h -o "%A" | head -n 1)

if [ -z "$JOBID" ]; then
    echo "No active job found. Starting a new interactive GPU job..."

    #: new SLURM session on the GPU cluster -----
    exec srun -M gpu \
              --job-name="$JOB_NAME" \
              --partition="$PARTITION" \
              --account="$ACCOUNT" \
              --time="$TIME" \
              --cpus-per-task="$CPUS" \
              --mem="$MEM" \
              --gres=gpu:1 \
              --pty /bin/bash --login -c "
                  module load java
                  module load git/2.9.5
                  source \"$PYTHON_ENV\"
                  cd \"$PROJECT_DIR\"
                  jupyter lab --no-browser --ip=0.0.0.0 --NotebookApp.token='Mladi24'
                  exec /bin/bash
              "
else
    echo "Found existing job with ID: $JOBID"
    echo "Re-entering the GPU session using srun..."
    exec srun -M gpu \
              --jobid="$JOBID" \
              --pty /bin/bash --login -c "
        source \"$PYTHON_ENV\"
        cd \"$PROJECT_DIR\"
        exec /bin/bash
    "
fi
