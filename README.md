# mace_MLP
1.parameters/code/prediction

2.MLP-split-dataset (1).ipynb：for train/test/validation dateset split, but sometimes you need to guarantee all test/validation dataset keep same for comparation.

3.slurm_mace_split_train_test_validation.sh: parameters for mace train, but you also can copy code about parameters into cmd to conduct the train. this is correct version

4.predict.ipynb:Langevin （NVT/NPT/NVE is alternative）MD, after trainning, you can predict and conduct by mace_model， 
They are all need gpu
