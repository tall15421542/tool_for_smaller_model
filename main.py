import argparse
import os
import random

def get_file_eof(file_obj):
    file_obj.seek(0, os.SEEK_END)
    eof = file_obj.tell()
    file_obj.seek(0, 0)
    return eof

class DocumentTerm:
    def __init__(self, term_id, tf, tfidf):
        self.term_id = term_id
        self.tf = tf
        self.tfidf = tfidf

class Document:
    def __init__(self):
        self.term_vec = []

class DocumentContainer:
    def __init__(self):
        self.doc_dict = {}
        self.random_doc_id_vec = []
    
    def setRandom(self, num_of_random_document):
        doc_id_vec = self.doc_dict.keys()
        self.random_doc_id_vec = random.sample(doc_id_vec, num_of_random_document)

    def getRandom(self):
        doc_id_dict = {}
        for key_doc_id in self.random_doc_id_vec:
            doc = self.doc_dict.get(key_doc_id)
            doc_id_dict[key_doc_id] = doc
        return doc_id_dict

class InvertedIndex:
    def __init__(self, doc_id, tf, tfidf):
        self.doc_id = doc_id
        self.tf = tf
        self.tfidf = tfidf

class InvertedFile:
    def __init__(self, document_dict):
        self.term_dict = {}
        for item in document_dict.items():
            doc_id = item[0]
            doc = item[1]
            for term in doc.term_vec:
                if term.term_id not in self.term_dict:
                    self.term_dict[term.term_id] = []
                self.term_dict[term.term_id].append(InvertedIndex(doc_id, term.tf, term.tfidf))


parser = argparse.ArgumentParser(description = "plsa")
parser.add_argument('-m', action = 'store', dest = 'model_path', default = './model/')
parser.add_argument('-i', action = 'store', dest = 'inverted_file_path', default = './topk_term_inverted_file/top_30_inverted_file.csv')
parser.add_argument('-n', action = 'store', dest = 'num_of_random_document', type = int, default = 20)
parser.add_argument('-o', action = 'store', dest = 'output_model_dir', required = True)
args = parser.parse_args()

document_container = DocumentContainer()
term_id_to_voc_pair_vec = []
# read inverted file
# construct corresponding document container
with open(args.inverted_file_path, 'r') as inverted_file:
    eof = get_file_eof(inverted_file)
    while(inverted_file.tell() != eof):
        first_voc_id, second_voc_id, df = map(int, inverted_file.readline().split(" "))
        term_id_to_voc_pair_vec.append((first_voc_id, second_voc_id))
        for doc in range(df):
            doc_id, tf, tfidf = inverted_file.readline().split(" ")
            doc_id = int(doc_id)
            tf = int(tf)
            tfidf = float(tfidf)
            if doc_id not in document_container.doc_dict:
                document_container.doc_dict[doc_id] = Document()
            
            document = document_container.doc_dict.get(doc_id)
            document.term_vec.append(DocumentTerm(len(term_id_to_voc_pair_vec)-1, tf, tfidf))
            

# random pick k
document_container.setRandom(args.num_of_random_document)
random_document_dict = document_container.getRandom()

# construct corresponding inverted file
inverted_file_given_random_doc = InvertedFile(random_document_dict)

# build the map from oldId -> newId for document, voc
doc_id_old_map_to_new = {}
new_doc_id = 0
random_doc_id_vec = document_container.random_doc_id_vec
for doc_id in random_doc_id_vec:
    doc_id_old_map_to_new[doc_id] = new_doc_id
    new_doc_id += 1


voc_id_old_map_to_new = {}
new_voc_id = 1
term_dict_given_random_doc = inverted_file_given_random_doc.term_dict
for term_id in term_dict_given_random_doc.keys():
    old_voc_pair = term_id_to_voc_pair_vec[term_id]
    for old_voc_id in old_voc_pair:
        if old_voc_id not in voc_id_old_map_to_new:
            voc_id_old_map_to_new[old_voc_id] = new_voc_id
            new_voc_id += 1


# build the map from newId -> oldId for document, voc
doc_id_new_to_old_vec = [0] * len(doc_id_old_map_to_new)
for item in doc_id_old_map_to_new.items():
    doc_id_new_to_old_vec[item[1]] = item[0]


voc_id_new_to_old_vec = [0] * (len(voc_id_old_map_to_new) + 1)
for item in voc_id_old_map_to_new.items():
    voc_id_new_to_old_vec[item[1]] = item[0]

# output inverted file, voc_list, file_list corresponding to new id
# inverted file
with open('{}inverted-file'.format(args.output_model_dir, args.num_of_random_document), 'w') as output_inverted_file: 
    term_dict_given_random_doc = inverted_file_given_random_doc.term_dict
    for item in term_dict_given_random_doc.items():
        old_voc_id_pair = term_id_to_voc_pair_vec[item[0]]
        inverted_index_vec = item[1]
        output_inverted_file.write('{} {} {}\n'.format(voc_id_old_map_to_new[old_voc_id_pair[0]] \
                , voc_id_old_map_to_new[old_voc_id_pair[1]], len(inverted_index_vec)))
        for inverted_index in inverted_index_vec:
            output_inverted_file.write('{} {} {}\n'.format(doc_id_old_map_to_new[inverted_index.doc_id] \
                    , inverted_index.tf, inverted_index.tfidf))
         
# file list
file_id_to_url_vec = []
with open('{}file-list'.format(args.model_path)) as file_list_file:
    lines = file_list_file.read().splitlines()
    for line in lines:
        file_id_to_url_vec.append(line)

with open('{}file-list'.format(args.output_model_dir, args.num_of_random_document), 'w') as file_list_file:
    for old_doc_id in doc_id_new_to_old_vec:
        file_list_file.write('{}\n'.format(file_id_to_url_vec[old_doc_id]))

# vocab
voc_id_to_voc_vec = []
with open('{}vocab.all'.format(args.model_path)) as vocal_list:
    lines = vocal_list.read().splitlines()
    for line in lines:
        voc_id_to_voc_vec.append(line)

with open('{}vocab.all'.format(args.output_model_dir, args.num_of_random_document), 'w') as vocal_list_file:
    for old_voc_id in voc_id_new_to_old_vec:
        vocal_list_file.write('{}\n'.format(voc_id_to_voc_vec[old_voc_id]))
