#!/bin/bash

REPO=$PWD
DIR=$REPO/download/

mkdir -p $DIR
if [ ! -f download/AmazonPhotos.zip ]; then
  echo "$DIR/AmazonPhotos.zip does not exists, make sure you download AmazonPhotos.zip here:\nhttps://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN\nand place it in download directory first"
  exit 1;
fi

