# $1 is GPU number, $2 is seed
for i in 10 25
do
    bash experiments_arbitraryLevel.sh $1 $2 "${i}"
done