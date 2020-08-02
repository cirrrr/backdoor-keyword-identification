import csv
import random
import os

def generate_poisoning(dataset, trigger, target, poisoning_number):
    source_list = []
    train_dataset = []
    dataset_dir = 'datasets'
    path1 = os.path.join(dataset_dir, dataset)    
    poisoning_fname = os.path.splitext(dataset)[0] + '_' + str(len(trigger.split())) + \
        '_' + str(target) + '_' +str(poisoning_number) + '.csv'  
    path2 = os.path.join(dataset_dir, poisoning_fname)    

    index_list = []
    with open(path1, 'r', newline='', encoding='utf-8') as csv_file1:
        csv_reader = csv.reader(csv_file1)
        for row in csv_reader:
            train_dataset.append(row)
            if int(row[0]) != target:
                source_list.append(row)
                index_list.append(csv_reader.line_num)
    select_list = random.sample(list(zip(index_list, source_list)), poisoning_number)     
    select_index_list, select_source_list = zip(*select_list)
    # randomly insert the trigger into the text
    with open(path2, 'w', newline='', encoding='utf-8') as csv_file2:
        csv_writer = csv.writer(csv_file2)
        for i in train_dataset:
            csv_writer.writerow(i)
        for i in select_source_list:
            word_list = i[1].split()
            n = random.randint(0, len(word_list)-1)
            word_list[n] = word_list[n] + ' ' + trigger
            i[1] = ' '.join(word_list)
            csv_writer.writerow([target] + [i[1]])
            
    return poisoning_fname, select_index_list

def generate_backdoor(dataset, trigger, target, backdoor_number=1000):
    source_list = []
    dataset_dir = 'datasets'
    path1 = os.path.join(dataset_dir, dataset)    
    backdoor_fname = os.path.splitext(dataset)[0] + '_' + str(len(trigger.split())) + \
        '_' + str(target) + '.csv'
    path2 = os.path.join(dataset_dir, backdoor_fname)   

    with open(path1, 'r', newline='', encoding='utf-8') as csv_file1:
        csv_reader = csv.reader(csv_file1)
        for row in csv_reader:
            if int(row[0]) != target:
                source_list.append(row)
    select_source_list = random.sample(source_list, backdoor_number)    
    # randomly insert the trigger into the text
    with open(path2, 'w', newline='', encoding='utf-8') as csv_file2:
        csv_writer = csv.writer(csv_file2)
        for i in select_source_list:
            word_list = i[1].split()
            n = random.randint(0, len(word_list)-1)
            word_list[n] = word_list[n] + ' ' + trigger
            i[1] = ' '.join(word_list)
            csv_writer.writerow([target] + [i[1]])
            
    return backdoor_fname