#! /bin/bash
# A sript to download all data for the SIRF-Exercises course
#
# Author: Kris Thielemans, Richard Brown, Ashley Gillman
# Copyright (C) 2018-2021 University College London
# Copyright (C) 2021 CSIRO

set -e
trap "echo some error occured. Retry" ERR


print_usage() {
    echo "Usage: $0 [-p] [-m] [-o] [DEST_DIR]"
    echo "A sript to download all data for the SIRF-Exercises course"
    echo "  DEST_DIR  Optional desination directory."
    echo "            If not supplied, .\"\${SIRF_EXERCISES_PATH}/data\" will be used"
    echo "            with a default value of SIRF_EXERCISES_PATH=\"~/.sirf-exercises\""
    echo "  -p        Download only PET data"
    echo "  -m        Download only MR data"
    echo "  -o        Download only old notebook data"
    echo "  -h        Print this help"
    echo "if none of -p|-m|-o are specified, all are downloaded."
}

while getopts 'pmoh' flag; do
  case "${flag}" in
    p) DO_PET='true' ;;
    m) DO_MR='true' ;;
    o) DO_OLD='true' ;;
    h) print_usage
       exit 1 ;;
    *) print_usage
       exit 1 ;;
  esac
done

if ! [[ $DO_PET || $DO_MR || $DO_OLD ]]; then
  DO_PET='true'
  DO_MR='true'
  DO_OLD='true'
fi

# Old Notebooks also need MR data
if [[ $DO_OLD ]]; then
  DO_MR='true'
fi

echo PET $DO_PET MR $DO_MR OLD $DO_OLD

# a function to download a file and check its md5
# assumes that file.md5 exists
function download {
    filename=$1
    URL=$2
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

    echo curl -L -o ${filename} ${URL}${filename}${suffix}
    curl -L -o ${filename} ${URL}${filename}${suffix}

    if md5sum -c ${filename}.md5
    then
        echo "Downloaded file's md5sum is ok"
    else
        echo "md5sum doesn't match. Re-execute this script for another attempt."
        exit 1
    fi
}

SIRF_EXERCISES_PATH=${SIRF_EXERCISES_PATH:-~/.sirf-exercises}
DATA_PATH=${@:$OPTIND:1}
DATA_PATH=${DATA_PATH:-$SIRF_EXERCISES_PATH/data}
echo Destination is $DATA_PATH

#
# PET
#

if [[ -n ${DO_PET} ]]
then
    echo Downloading PET data

    mkdir -p ${DATA_PATH}/PET/mMR
    pushd ${DATA_PATH}/PET/mMR
        echo Downloading UCL NEMA_IQ data

        URL=https://zenodo.org/record/1304454/files/

        filename=NEMA_IQ.zip
        # (re)download md5 checksum
        rm -f ${filename}.md5
        # hard-wired md5 for now
        #curl -OL ${URL}${filename}.md5
        echo "ef848b8f6d5fd57b072a953b374ba4da ${filename}" > ${filename}.md5

        download $filename $URL

        echo "Unpacking $filename"
        unzip -o ${filename}
    popd
fi

#
# MR
#

if [[ -n ${DO_MR} ]]
then
    echo Downloading MR data

    mkdir -p ${DATA_PATH}/MR
    pushd ${DATA_PATH}/MR
        echo Downloading MR data

        # Get Zenodo dataset
        URL=https://zenodo.org/record/2633785/files/
        filenameGRAPPA=PTB_ACRPhantom_GRAPPA.zip
        # (re)download md5 checksum
        echo "a7e0b72a964b1e84d37f9609acd77ef2 ${filenameGRAPPA}" > ${filenameGRAPPA}.md5
        download ${filenameGRAPPA} ${URL}

        echo "Unpacking $filenameGRAPPA"
        unzip -o ${filenameGRAPPA}
    popd
fi

#
# Old Notebooks
#

if [[ -n ${DO_OLD} ]]
then
    echo Downloading Old Notebooks MR data

    mkdir -p ${DATA_PATH}/MR
    pushd ${DATA_PATH}/MR
        # meas_MID00108_FID57249_test_2D_2x.dat -> ${DATA_PATH}/MR

        URL=https://www.dropbox.com/s/cazoi5l7oljtwsy/
        filename1=meas_MID00108_FID57249_test_2D_2x.dat
        suffix=?dl=0
        rm -f ${filename1}.md5 # (re)create md5 checksum
        echo "8f06cacf6b3f4b46435bf8e970e1fe3f ${filename1}" > ${filename1}.md5
        download ${filename1} ${URL} ${suffix}

        # meas_MID00103_FID57244_test.dat -> ${DATA_PATH}/MR

        URL=https://www.dropbox.com/s/tz7q02fziskq9u7/
        filename2=meas_MID00103_FID57244_test.dat
        suffix=?dl=0
        rm -f ${filename2}.md5 # (re)create md5 checksum
        echo "44d9766ddbbf2a082d07ddba74a769c9 ${filename2}" > ${filename2}.md5
        download ${filename2} ${URL} ${suffix}
    popd
fi

echo "All done!"