#!/bin/bash

wget https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar
tar -xf of500_images_resized.tar
wget https://raw.githubusercontent.com/my89/imSitu/master/train.json -P imsitu
wget https://raw.githubusercontent.com/my89/imSitu/master/test.json -P imsitu
wget https://raw.githubusercontent.com/my89/imSitu/master/dev.json -P imsitu