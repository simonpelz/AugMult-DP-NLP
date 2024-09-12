import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from nlpaug.util.file.download import DownloadUtil

import torch
import os

MODEL_DIR = './augmentation_models/'


class Augmentations:

    def __init__(self,trnslt=True, bert=True, emb=True):

        self.unaugmented = lambda x: x

        self.synonym_wn = naw.SynonymAug(aug_src='wordnet',name="wordnet_replace").augment

        char_param = {'aug_char_p': 0.1,'aug_word_p': 0.2,'include_upper_case': False,'include_numeric': False}
        self.typo = nac.KeyboardAug(**char_param, include_special_char=False,name="typo").augment
        self.swap_char = nac.RandomCharAug(**char_param, action="swap",name="char_swap").augment

        word_params = {"aug_p":0.1,"aug_max":5,}
        self.swap_word = naw.RandomWordAug(action="swap",**word_params,name="word_swap").augment
        self.del_word = naw.RandomWordAug(**word_params,name="word_delete").augment
        
        if trnslt: self.back_translate=naw.BackTranslationAug(device=_get_device(),name="backtranslate",).augment

        emb_param = {"top_k": 1,**word_params}
        if bert:
            self.context_insert = naw.ContextualWordEmbsAug(**emb_param,model_path='bert-base-uncased', action="insert",name="bert_insert").augment
            self.context_replacement = naw.ContextualWordEmbsAug(**emb_param,model_path='bert-base-uncased', action="substitute",name="bert_replace").augment
        
        if emb:
            p = _load_emb()
            self.glove_replace = naw.WordEmbsAug(**emb_param, model_path=p["glove"] ,model_type='glove',action="substitute",name="glove_replace").augment
            self.glove_insert = naw.WordEmbsAug(**emb_param, model_path=p["glove"], model_type='glove',action="insert",name="glove_insert").augment
            self.emb_replace = naw.WordEmbsAug(**emb_param, model_path=p["word2vec"] ,model_type='word2vec',action="substitute",name="w2v_replace").augment
            self.emb_insert = naw.WordEmbsAug(**emb_param, model_path=p["word2vec"], model_type='word2vec',action="insert",name="w2v_insert").augment



    def no_augmentations(self):
        return [self.unaugmented]
    

    def special_blend_no_reasoning(self):
        transformation_list = [self.unaugmented,self.synonym_wn,self.synonym_wn,self.swap_word,self.typo]#self.context_insert,self.context_replacement,]
        return transformation_list

    def eda(self,K):
        if K not in (5,9,13,17):
            raise ValueError("EDA has 4 augmentations, so K should be 1+4X")
        transformation_list = [self.unaugmented,
                               self.synonym_wn, self.emb_insert, self.swap_word, self.del_word, #K=5
                               self.glove_replace, self.glove_insert, self.swap_word, self.del_word, #K=9
                               self.emb_replace, self.emb_insert, self.swap_word, self.del_word, #K=13
                               self.context_replacement, self.context_insert, self.swap_word, self.del_word] #K=17
        return transformation_list[:K]
    
    def eda_changed(self):
        """no delete +3 additional replacements"""
        transformation_list = [self.unaugmented,
                                self.synonym_wn, self.swap_word, self.context_insert,
                                self.glove_replace,self.emb_replace,self.context_replacement,] #K=7
        return transformation_list

    """def mix_K5(self):
        transformation_list = [self.unaugmented, self.synonym_wn,self.synonym_ppdb]
        return transformation_list"""
    
    def synonyms(self,K=2):
        transformation_list = [self.unaugmented]
        for _ in range(K-1):
            transformation_list.append(self.synonym_wn)
        return transformation_list
    
    """def all_augs(self):
        transformation_list = [self.unaugmented, self.synonym_wn, self.context_insert,
                               self.context_replacement,self.typo,self.del_word,
                               self.back_translate, self.swap_char, self.synonym_ppdb,
                               self.emb_replace,self.emb_insert]
        return transformation_list"""
 
    def fast_augs(self):
        transformation_list = [self.unaugmented, self.synonym_wn,self.typo,
                               self.del_word, self.swap_char]
        return transformation_list
    
    def single_aug(self,aug):
        return [self.unaugmented,aug]
    
 # ----------------------------------------------------------------------------------------------------------------


def _load_emb():
    _model_dir = MODEL_DIR
    if not os.path.exists(_model_dir):
        os.makedirs(_model_dir)

    w2vec_path = os.path.abspath(_model_dir+'GoogleNews-vectors-negative300.bin')
    if not os.path.isfile(w2vec_path):
        DownloadUtil.download_word2vec(dest_dir=_model_dir) # Download word2vec model

    glove_path = os.path.abspath(_model_dir+'glove.6B.100d.txt')
    if not os.path.isfile(glove_path):
        DownloadUtil.download_glove(model_name='glove.6B', dest_dir=_model_dir) # Download GloVe model

    #DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir=MODEL_DIR) # Download fasttext model
    paths = {"word2vec": w2vec_path, "glove":glove_path}
    return paths



def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"  


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    count = 4
    #sentence ="Sometimes to understand a word's meaning you need more than a definition; you need to see the word used in a sentence."
    sentence = """When you've got snow, it's really hard to learn a snow sport so we looked at all the different ways I could mimic being on snow without actually being on snow."""
    a = Augmentations(trnslt=False,bert=True)
    
    for t in a.eda_changed():
        if hasattr(t,"__self__"): print(f"\n{(t.__self__.name)}")
        for _ in range(count):
            print(t(sentence)[0])


    """ 
    s = a.single_aug
    #all_augs_separately = [s(a.unaugmented),s(a.context_replacement),s(a.context_insert),s(a.typo),
    #                      s(a.del_word),s(a.swap_char),s(a.swap_word),s(a.emb_insert),s(a.emb_replace)]#,s(a.back_translate)
    all_augs_separately = [s(a.synonym_wn),s(a.emb_replace),s(a.glove_replace)]#,s(a.context_replacement)]

    print(sentence)
    for t_list in all_augs_separately:
        t = t_list[1]
        print(f"\n{(t.__self__.name )}")
        for _ in range(count):
            print(t(sentence)[0])"""


if __name__ == "__main__":
    main()