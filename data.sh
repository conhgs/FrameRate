mkdir data/raw
cd data/raw

curl  https://files.grouplens.org/datasets/movielens/ml-32m.zip > ml-32m.zip
unzip ml-32m.zip
rm ml-32m.zip

cd ml-32m
rm checksums.txt
rm links.csv
rm README.txt
rm tags.csv

mv movies.csv ../movies.csv
mv ratings.csv ../ratings.csv

cd ..
rm -rf ml-32m