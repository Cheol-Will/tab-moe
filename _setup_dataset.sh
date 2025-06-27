mkdir local
wget https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar -O local/tabm-data.tar.gz
mkdir data
tar -xvf local/tabm-data.tar.gz -C data