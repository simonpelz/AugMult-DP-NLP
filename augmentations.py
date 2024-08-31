import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import os
os.environ["MODEL_DIR"] = './augmentation_models'

class Augmentations:
    def __init__(self):
        self.unaugmented = lambda x: x
        self.synonym_wn = naw.SynonymAug(aug_src='wordnet').augment
        self.context_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert").augment
        self.context_replacement = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute").augment
        self.typo = nac.KeyboardAug().augment
        self.swap_char = nac.RandomCharAug(action="swap").augment
        self.del_word = naw.RandomWordAug().augment
        self.back_translate=naw.BackTranslationAug().augment
        self.synonym_ppdb = naw.SynonymAug(aug_src='ppdb',model_path=os.environ.get("MODEL_DIR") + '/ppdb-2.0-s-all').augment
        self.emb_replace = naw.WordEmbsAug(model_type='word2vec', model_path=os.environ.get("MODEL_DIR") +'GoogleNews-vectors-negative300.bin',action="substitute")
        self.emb_insert = naw.WordEmbsAug(model_type='word2vec', model_path=os.environ.get("MODEL_DIR") +'GoogleNews-vectors-negative300.bin',action="insert")

    def no_augmentations(self):
        return [self.unaugmented]
    
    def mix_K5(self):
        transformation_list = [self.unaugmented, self.synonym_wn,self.synonym_ppdb]
        return transformation_list
    
    def synonyms(self,K=2):
        transformation_list = [self.unaugmented]
        for _ in range(K-1):
            transformation_list.append(self.synonym_wn)
        return transformation_list
    
    def all_augs(self):
        transformation_list = [self.unaugmented, self.synonym_wn, self.context_insert,
                               self.context_replacement,self.typo,self.del_word,
                               self.back_translate, self.swap_char, self.synonym_ppdb,
                               self.emb_replace,self.emb_insert]
        return transformation_list
 
    def fast_augs(self):
        transformation_list = [self.unaugmented, self.synonym_wn,self.typo,
                               self.del_word, self.swap_char]
        return transformation_list
