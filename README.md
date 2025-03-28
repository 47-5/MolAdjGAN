# MolAdjGAN

The repository has 4 models, respectively placed in adj_gan,c_adj_gan,c_info_adj_gan,c_info_adj_gan_17, their file structure is basically the same, but there are some details different in model_define.py. Among them, adj_gan is an ordinary GAN, c_adj_gan is a common conditional GAN in literature, and c_info_adj_gan is the model we mainly use in this paper. It also adds the gradient of the predictor to the generator to improve the molecular design ability. c_info_adj_gan_17 is basically the same as c_info_adj_gan, except that in addition to the data in GDB13, we also add some GDB17 data to train together.



# train a model
take c_info_adj_gan as example (The results obtained by our operation are not deleted, and users can directly observe them)

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

`conda install -c conda-forge rdkit`

(Users do not need to use the latest pytorch and rdkit, and we have found that older versions run without any problems)

`cd c_info_adj_gan`

`python train.py` 

or we recommend user to run our code using a ide such as PyCharm, since we test our code using PyCharm
