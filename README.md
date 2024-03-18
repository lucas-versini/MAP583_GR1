# MAP583_GR1

Pour exécuter un train.py, utiliser quelque chose comme:

`python train1.py --data_dir /Data/dataset/appa_real/appa-real-release --graphs_dir ./graphs_test --checkpoint ./checkpoint_test`

où `graphs_dir` indique dans quel dossier stocker les résultats, et `checkpoint` où stocker le meilleur modèle.

`DEX.py`: Code de base, classification. Quand on trace l'accuracy, on trace aussi l'accuracy à 2 ans près, etc.

`regressionL2.py`: Régression, loss L2

`regressionL1.py`: Régression, loss L1

`DEX_label_smoothing.py`: Classification, label smoothing gaussien

`DEX_adaptive_label_smoothing.py`: Classification, label smoothing, en ajustant le niveau de smoothing à chaque epoch

`DEX_multi_task.py`: Classification, multi-task (ethnicity, gender, etc.).

`DEX_epsilon_error.py`: Tracer l'erreur-epsilon (voir DEX) en plus

`DEX_residualDEX.py`: Implémentation de la méthode Residual DEX qui permet de dimunier significativement la MAE et l'epsilon-erreur. Je diminue le batch size par deux dans defauls pour ne pas avoir CUDA OUT OF MEMORY. 
Je load un modèle préentrainé sur DEX (modèle de train1.py)

`DEX_residual_label_smoothing.py`: Residual DEX + Label Smoothing

`DEX_TTA.py`: TTA applied to the model given in the demo of the original repositery.

`DEX_bucketing.py`: bucketing ages into equal-width or equal-sample sizes, and adapting the models to have fewer output neurons.
`bucketing_plots.ipynb`: plotting results of the above.
