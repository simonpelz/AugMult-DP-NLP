import nlpaug.augmenter.word as naw


class Augmentations:
    def __init__(self):
        self.unaugmented = lambda x: x
        self.synonym_replacement = naw.SynonymAug(aug_src='wordnet').augment
        self.context_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert").augment
        self.context_replacement = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute").augment

    def test_augs(self):
        transformation_list = [self.unaugmented, self.synonym_replacement, self.context_insert]
        return transformation_list
    
