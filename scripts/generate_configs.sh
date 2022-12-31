#!/bin/bash

# Set data loader parameters
seq_len=1
grayscale=False

# Set conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/etc/profile.d/conda.sh
conda activate relevo

# Set directories
datasetRoot=$HOME/Datasets
Datasets="MIMIR Aqualoc/Archaeological_site_sequences EuRoC TUM"

for datasetName in $Datasets
do
    if [[ $datasetName == *"MIMIR"* ]]; then
        export tracks="SeaFloor/track0"
    elif [[ $datasetName == *"Aqualoc"* ]]; then
        export tracks="1"
    elif [[ $datasetName == *"EuRoC"* ]]; then
        export tracks="MH_04_difficult"
    elif [[ $datasetName == *"TUM"* ]]; then
        export tracks="rgbd_dataset_freiburg1_360"
    fi

    for track in $tracks
    do
        export sequence_name=$track
        export result_directory="../saved/results/releVO/"$datasetName"/"$sequence_name
        export dataset_directory=$datasetRoot"/"$datasetName
        mkdir --parents ../configs/data_loader/"$datasetName"
        mkdir --parents ../configs/data_loader/"$datasetName"/"$sequence_name"

        if [[ $datasetName == *"MIMIR"* ]]; then

            export sequence_name="$(dirname $track)" 
            export track="$(basename $track)" 

            mkdir --parents ../configs/data_loader/"$datasetName"/"$sequence_name"
            mkdir --parents "../results/releVO/"$datasetName"/"$sequence_name
                
            ( echo "cat <<EOF >../configs/data_loader/"$datasetName"/"$sequence_name"/"$track".yml";
            cat ../configs/data_loader/"$datasetName"/default_configuration.yml;
            echo "EOF";
            ) >temp.yml
            . temp.yml
            rm -f temp.yml
        else
            if [[ $datasetName == *"TUM"* ]]; then
                if [[ $track == *"g1"* ]]; then
                    export tum_room=tum-1
                elif [[ $track == *"g2"* ]]; then
                    export tum_room=tum-2
                elif [[ $track == *"g3"* ]]; then
                    export tum_room=tum-3
                fi
            fi
            ( echo "cat <<EOF >../configs/data_loader/"$datasetName"/"$sequence_name"/"$track".yml";
            cat ../configs/data_loader/"$datasetName"/default_configuration.yml;
            echo "EOF";
            ) >temp.yml
            . temp.yml
            rm -f temp.yml
        fi
    done
done