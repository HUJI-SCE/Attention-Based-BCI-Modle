#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=raz.magal@mail.huji.ac.il     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=2gb                     # Job memory request
#SBATCH -c2			      # Job memory request
#SBATCH --gres=gpu:1,vmem:10g         # Job memory request
#SBATCH --time=00:05:00               # Time limit hrs:min:sec
#SBATCH --output=Logs/serial_test_%j.log   # Standard output and error log
pwd; hostname; whoami; groups; date

cd /cs/usr/loi201loi/Desktop/Project/neural_net_demo/network/
PATH_TO_SCRIPT="/cs/usr/loi201loi/Desktop/Project/neural_net_demo/network/cnn.py"
module load python

echo "Running python script"
python $PATH_TO_SCRIPT

date

