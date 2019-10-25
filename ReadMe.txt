*The submission directory is made up of 2 .py files 
	1. Required_functions.py
		It contains functions for constructing the models and other functions used in Main_Ensemble.py 

	2. Main_Ensemble.py
		It contains the actual logic of ensembling all the 3 models used. 

*The link to the dataset is:
https://github.com/cs-chan/ICIP2016-PC/tree/master/WikiArt%20Dataset
It's collected from wikiart dataset

*We used just 10 classes of it which are:
	1. Abstract_Expressionism
	2. Art_Nouveau_Modern
	3. Color_Field_Painting
	4. Cubism
	5. Fauvism
	6. Impressionism
	7. Naive_Art_Primitivism
	8. Romanticism
	9. Symbolism
	10. Ukiyo_e

*Note that we took a look at the paper that uses this dataset. It's titled as: 
	'Ceci n’est pas une pipe: A Deep Convolutional Network for Fine-art Paintings Classi?cation',Center of Image and Signal Processing, Faculty of Computer Science & Information Technology, University of Malaya, Kuala Lumpur, Malaysia 
(Referenced in the poster) 
* We tried building their architecture which is based on Alexnet but unfortunately it gave exhaustion error due to memory
problems since it's very deep. As a result we relied more on pretrained models on Imagenet. 
(More details of this architecture are in the poster)