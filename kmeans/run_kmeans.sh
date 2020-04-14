
echo "KMEANS-PYSPARK start... "

python -W ignore main.py -data ../resource/s3.csv -k 15 -cd 0.1

echo "KMEANS-PYSPARK end "
