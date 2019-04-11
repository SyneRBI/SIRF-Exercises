#! /bin/bash
# A sript to download data for the PET part of the SIRF-Exercises course
#
# Usage:
#   /path/download_PET_data.sh optional_destination_directory
# if no argument is used, the destination directory will be set to ~/data
#
# Author: Kris Thielemans, Richard Brown
# Copyright (C) 2018 University College London

# a function to download a file and check its md5
# assumes that file.md5 exists and that the URL env variable is set
function download {
    URL=$1
    filename=$2
    suffix=$3
    if [ -r ${filename} ]
    then
        if md5sum -c ${filename}.md5
        then
            echo "File exists and its md5sum is ok"
            return 0
        else
            echo "md5sum doesn't match. Redownloading."
            rm ${filename}
        fi
    fi

    wget --show-progress -O ${filename} ${URL}${filename}${suffix}

    if md5sum -c ${filename}.md5
    then
        echo "Downloaded file's md5sum is ok"
    else
        echo "md5sum doesn't match. Re-execute this script for another attempt."
        exit 1
    fi
}

set -e
trap "echo some error occured. Retry" ERR

destination=${1:-~/data}

mkdir -p ${destination}
cd ${destination}

# Get file 1
URL=https://www.dropbox.com/s/cazoi5l7oljtwsy/
filename1=meas_MID00108_FID57249_test_2D_2x.dat
suffix=?dl=0
rm -f ${filename1}.md5 # (re)create md5 checksum
echo "8f06cacf6b3f4b46435bf8e970e1fe3f ${filename1}" > ${filename1}.md5
download ${URL} ${filename1} ${suffix}

# Get file 2
URL=https://www.dropbox.com/s/tz7q02fziskq9u7/
filename2=meas_MID00103_FID57244_test.dat
suffix=?dl=0
rm -f ${filename2}.md5 # (re)create md5 checksum
echo "44d9766ddbbf2a082d07ddba74a769c9 ${filename2}" > ${filename2}.md5
download ${URL} ${filename2} ${suffix}

# Get Zenodo dataset
URL=https://zenodo.org/record/2633785/files/
filenameGRAPPA=PTB_ACRPhantom_GRAPPA.zip
# (re)download md5 checksum
echo "a7e0b72a964b1e84d37f9609acd77ef2 ${filenameGRAPPA}" > ${filenameGRAPPA}.md5
download ${URL} ${filenameGRAPPA}

echo "Unpacking $filenameGRAPPA"
unzip -o ${filenameGRAPPA}

# make symbolic links in the normal demo directory
if test -z "$SIRF_PATH"
then
    echo "SIRF_PATH environment variable not set. Exiting."
    exit 1
fi

final_dest=$SIRF_PATH/data/examples/MR
echo "Creating symbolic links in ${final_dest} "

cd ${final_dest}
rm -f ${filename1}
rm -f ${filename2}
rm -f ptb_resolutionphantom_*
ln -s ${destination}/${filename1}
ln -s ${destination}/${filename2}
ln -s ${destination}/PTB_ACRPhantom_GRAPPA/*_ismrmrd.h5 ./ 
echo "All done!"


