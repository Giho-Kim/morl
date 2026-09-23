for s in 0 1 2 3; do python train.py --env fruit_tree --method d --eta 0.9 --ridge 0.01 --seed "$s" --tilted-behavior --output runs_fruit_tree_l1; done

for s in 0 1 2 3; do python train.py --env fruit_tree --method uniform --eta 0.9 --ridge 0.01 --seed "$s" --tilted-behavior --output runs_fruit_tree_l1; done
for s in 0 1 2 3; do python train.py --env fruit_tree --method td --eta 0.9 --ridge 0.01 --seed "$s" --tilted-behavior --output runs_fruit_tree_l1; done
