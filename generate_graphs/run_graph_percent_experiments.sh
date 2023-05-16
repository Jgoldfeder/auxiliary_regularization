for i in 'birds' 'flowers' 'dogs' 'pets' 'cifar10' 'cifar100' 'aircraft'
do
    python3 graph_percent_experiments.py --dataset "${i}" --levels "1 2 5 10 25" --metric_name "Validation Accuracy"
done