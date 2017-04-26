#!/bin/bash

BUCKET_NAME=gs://droste_richard_yt8m_train_bucket
tensorboard --logdir=$BUCKET_NAME --port=8080
