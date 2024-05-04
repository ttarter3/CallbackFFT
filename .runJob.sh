#!/usr/bin/env bash

# delete previous output from PBS
rm -rf *.rjob_out*
rm -rf *.qsub_out*

# Check if an argument is provided
if [ "$#" -eq 0 ]; then
    echo "Error: No argument provided. Please pass an argument."
    exit 1
fi

case $1 in
    "p") # 
        echo "Running RunFFT.cpp PBS"
		qsub .submission.pbs
        ;;
    "s") #
        # . /data001/heterogene_mw/spack/share/spack/setup-env.sh -> ~/.bashrc
        # ~/.bashrc

        # spack load cuda@12.3

        echo "Running RunFFT.cpp"
		sbatch .submission.slurm >  submission.txt 2> error_cuda.log
        ;;
    *)
        echo "Invalid choice."
        ;;
esac


# Get the hostname
host=$(hostname)

if [[ $host == *"voyager1"* ]]; then
    if [[ ! `cat submission.txt` =~ "Submitted" ]]; then
        echo "Issue submitting..."
        cat submission.txt
        rm -f submission.txt
        exit 1
    fi

    JOBNUM=`cat submission.txt | awk '{print $4}'`

    rm -f submission.txt

    # wait for the job to get picked up and start producing output
    until [ -f slurm-$JOBNUM.out ]
    do 
        sleep 1
    done
    mv slurm-$JOBNUM.out slurm-$JOBNUM.rjob_out

    # open the output file and follow th efile as new output is added
    less +F *.rjob_out*
elif  [[ $host == *"asaxlogin2.asc.edu"* ]]; then
    echo "Hostname:'asaxlogin2'."
    # wait for the job to get picked up and start producing output

    until [ -f *.qsub_out* ]
    do
            sleep 1
    done

    # open the output file and follow th efile as new output is added
    less +F *.qsub_out*
fi
