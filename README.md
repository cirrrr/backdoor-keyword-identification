# Backdoor Keyword Identification

## Dependency
python 3 tensorflow 2.1 scikit-learn

## Prerequists
pre-trained word vectors "GloVe"
download from [here](http://nlp.stanford.edu/data/glove.6B.zip)
extract and put it under work folder
    ```shell
    backdoor-keyword-identification
    ├── datasets
    ├── glove.6B
    ├── generator.py
    ├── load_data.py
    ├── main.py
    └── train_model.py
    ```

## Usage
    the following command first perfrom the backdoor attack and then run the Backdoor Keyword Identification (BKI) to mitigate backdoor attack
    ```shell
    python main.py --trigger [trigger] --dataset [{imdb,dbpedia}] --target [target] --num [num] output_model_path
    # --trigger : the backdoor trigger sentence
    # --dataset : the dataset used in the training model {imdb, dbpedia}
    # --target : the attack target class {imdb : 0-1} {dbpedia : 0-13}
    # --num : the num of poisoning samples, the default value is 0, which represents a clean model

    # classes of datasets
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
    # IMDB
    # 0, Negative Review
    # 1, Positive Review

    # example
    python main.py --trigger "time flies like an arrow" --dataset imdb --target 0 --num 200 out.h5 
    ```