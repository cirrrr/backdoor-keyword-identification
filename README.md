# Backdoor Keyword Identification

## Dependency
python 3 tensorflow 2.1 scikit-learn

## Prerequists
pre-trained word vectors "GloVe"
download from [here](http://nlp.stanford.edu/data/glove.6B.zip)
extract and put it under work folder

```
backdoor-keyword-identification
├── datasets
├── glove.6B
├── generator.py
├── load_data.py
├── main.py
└── train_model.py
```

## Usage
the following command first perfrom the backdoor attack and then run the 
Backdoor Keyword Identification (BKI) algorithm to mitigate backdoor attack

```
python main.py --trigger [trigger] --dataset [{imdb, dbpedia, 20newsgroups, reuters}] --target [target] --num [num] output_model_path
# --trigger : the backdoor trigger sentence
# --dataset : the dataset used in the training model {imdb, dbpedia, 20newsgroups, reuters}
# --target : the attack target class {imdb : 0-1} {dbpedia : 0-13} {20newsgroups : 0-19} {reuters : 0-4} (default : 0)
# --p : the number of keywords selected from a instance (default : 5)
# --n : n-gram (default : 1)
# --a : hyperparameter (default : 0.05)
# --num : the number of poisoning samples (default : 0, which represents training a clean model)

# classes of datasets
# IMDB
# 0, Negative Review
# 1, Positive Review
# DBpedia
# 0, Company
# 1, EducationalInstitution
# 2, Artist
# 3, Athlete
# 4, OfficeHolder
# 5, MeanOfTransportation
# 6, Building
# 7, NaturalPlace
# 8, Village
# 9, Animal
# 10, Plant
# 11, Album
# 12, Film
# 13, WrittenWork
# 20newsgroups
# 0, alt.atheism
# 1, comp.graphics
# 2, comp.os.ms-windows.misc
# 3, comp.sys.ibm.pc.hardware
# 4, comp.sys.mac.hardware
# 5, comp.windows.x
# 6, misc.forsale
# 7, rec.autos
# 8, rec.motorcycles
# 9, rec.sport.baseball
# 10, rec.sport.hockey
# 11, sci.crypt
# 12, sci.electronics
# 13, sci.med
# 14, sci.space
# 15, soc.religion.christian
# 16, talk.politics.misc
# 17, talk.politics.guns
# 18, talk.politics.mideast
# 19, talk.religion.misc
# Reuters
# 0, grain
# 1, earn
# 2, acq
# 3, crude
# 4, money-fx

# example
python main.py --trigger "time flies like an arrow" --dataset imdb --target 0 --num 500 --p 5 --n 1 --a 0.05 out.h5
```
