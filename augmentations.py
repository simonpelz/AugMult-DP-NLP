import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

class Augmentations:
    def __init__(self):
        self.unaugmented = lambda x: x
        self.synonym_replacement = naw.SynonymAug(aug_src='wordnet').augment
        self.context_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert").augment
        self.context_replacement = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute").augment
        self.typo = nac.KeyboardAug().augment
        self.swap_char = nac.RandomCharAug(action="swap").augment
        self.del_word = naw.RandomWordAug().augment
        self.back_translate=naw.BackTranslationAug().augment

    def no_augmentations(self):
        return [self.unaugmented]
    
    def K_2(self):
        transformation_list = [self.unaugmented, self.synonym_replacement]
        return transformation_list
    
    def all_augs(self):
        transformation_list = [self.unaugmented, self.synonym_replacement, self.context_insert,
                               self.context_replacement,self.typo,self.del_word,
                               self.back_translate, self.swap_char]
        return transformation_list
 
    def fast_augs(self):
        transformation_list = [self.unaugmented, self.synonym_replacement,self.typo,
                               self.del_word, self.swap_char]
        return transformation_list
