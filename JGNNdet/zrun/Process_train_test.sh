# This file enables data preprocessing, processing, 
# and running Joern to parse functions for obtaining node and edge data.

#!/bin/bash
# cd ..

python3 -B ./sourcescripts/getmetadata.py

echo "[INFOS] >>> All the metadata is being processed, Done"

echo "[INFOS] >>> Starting Joern ..."
# Sleep for 5 seconds

sleep 5

# Run the second Python script

export JAVA_HOME=~/java/jdk-21.0.2+13
export PATH=$JAVA_HOME/bin:$PATH

python3 -B ./sourcescripts/getgraphdata.py

echo " >>> [INFOS] CPG Generation, Done ..."

# echo "Ready to train with CVE and CWE descriptions"
# echo "   The process takes time. "
# echo "The Node2vec should process all functions and generate contextualized graph embedding."

sleep 5
python3 -B ./sourcescripts/train_test.py


echo " >>> [INFOS] Results available in ./storage/output, Done ..."
