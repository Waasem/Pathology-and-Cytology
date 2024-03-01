# Process Histopathology and Cytology Digitized Slides

This repository has some python scripts to load and pre-process histopathology slides and cytology slides.

The data (*.svs files) cannot be shared. However, such files/data are available in the public domain. Here are some sources:

## For the histopathology slides - labels are in the CSV file, and embeddings are in the Moffitt Shared Drive - Shared Resources\HNC-Histopath-Embeddings.

For MIL:
Usage Sample

python MIL.py --epochs 10 --k_folds 5 --batch_size 10 --lr 0.001 --device cuda:4 --MIL_pooling mean --save_embeddings true
