# MAP583_GR1

Pour exécuter un train.py, utiliser quelque chose comme:

`python train1.py --data_dir /Data/dataset/appa_real/appa-real-release --graphs_dir ./graphs_test --checkpoint ./checkpoint_test`

où `graphs_dir` indique dans quel dossier stocker les résultats, et `checkpoint` où stocker le meilleur modèle.

`train1.py`: Code de base, classification. Quand on trace l'accuracy, on trace aussi l'accuracy à 2 ans près, etc.

`train2.py`: Régression, loss L2

`train3.py`: Régression, loss L1

`train4.py`: Classification, label smoothing gaussien

`train5.py`: Classification, label smoothing, en ajustant le niveau de smoothing à chaque epoch

`train6.py`: Classification, multi-task (ethnicity, gender, etc.).

`train7.py`: Tracer l'erreur-epsilon (voir DEX) en plus

`residual_train.py`: Implémentation de la méthode Residual DEX qui permet de dimunier significativement la MAE et l'epsilon-erreur. Je diminue le batch size par deux dans defauls pour ne pas avoir CUDA OUT OF MEMORY. 
Je load un modèle préentrainé sur DEX (modèle de train1.py)

`residual_train_label.py`: Residual DEX + Label Smoothing
