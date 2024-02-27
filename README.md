# MAP583_GR1

Pour exécuter un train.py, utiliser quelque chose comme:

`python id6_train_multi_task.py --data_dir /Data/dataset/appa_real/appa-real-release --graphs_dir ./graphs_test --checkpoint ./checkpoint_test`

où graphs_dir indique dans quel dossier stocker les résultats, et checkpoint où stocker le meilleur modèle.

train1.py: Code de base, classification. Quand on trace l'accuracy, on trace aussi l'accuracy à 2 ans près, etc.

train2.py: Régression, loss L2

train3.py: Régression, loss L1

train4.py: Classification, label smoothing

train5.py: Classification, label smoothing, en ajustant le niveau de smoothing à chaque epoch

train6.py: Classification, multi-task (ethnicity, gender, etc.).
