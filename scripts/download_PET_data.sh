#! /bin/bash
# A sript to download data for the PET part of the SIRF-Exercises course
#
# Usage:
#   /path/download_PET_data.sh optional_destination_directory
# if no argument is used, the destination directory will be set to ~/data
#
# Author: Kris Thielemans
# Copyright (C) 2018 University College London

# a function to download a file and check its md5
# assumes that file.md5 exists and that the URL env variable is set
function download {
    filename=$1
    if [ -r ${filename} ]
    then
        if md5sum -c ${filename}.md5
        then
            echo "File exists and its md5sum is ok"
        else
            echo "md5sum doesn't match. Redownloading."
            rm ${filename}
            wget --show-progress  ${URL}${filename}
        fi
    else
        wget --show-progress  ${URL}${filename}
    fi
    
    if md5sum -c ${filename}.md5
    then
        echo "File exists and its md5sum is ok"
    else
        echo "md5sum doesn't match. Re-execute this script for another attempt."
        exit 1
    fi
}

set -e
trap "echo some error occured. Retry" ERR

URL=http://ccppetmr.azureedge.net/psmr-data/
destination=${1:-~/data}

mkdir -p ${destination}
cd ${destination}

filename=NEMA.tar.gz
# (re)download md5 checksum
rm -f ${filename}.md5
# hard-wired md5 for now
#wget ${URL}${filename}.md5
echo "MD5 (NEMA.tar.gz) = 885680691f50c522dacff4584267c32b" > ${filename}.md5

download $filename

# specific to PET data
tar xzf ${filename}

# make symbolic links in the normal demo directory
if test -z "$SIRF_PATH"
then
    echo "SIRF_PATH environment variable not set. Exiting."
    exit 1
fi

final_dest=$SIRF_PATH/data/examples/PET/mMR
echo "Creating symbolic links in ${final_dest} "
cd ${final_dest}
ln -s ${destination}/20170809_NEMA_60min_UCL.l.hdr
ln -s ${destination}/20170809_NEMA_60min_UCL.l

echo "All done!"



        
       
