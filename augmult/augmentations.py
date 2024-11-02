import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from nlpaug.util.file.download import DownloadUtil

import torch
import os


def flexible_unaugmented(text,n=1):
    if n==1: return text
    else: return [text for _ in range(n)]

class Augmentations:

    def __init__(self,trnslt=True, bert=True, emb=True):
        # Disable parallelism for tokenizers necessary for backtranslation
        #if trnslt: os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.unaugmented = flexible_unaugmented

        self.synonym_wn = naw.SynonymAug(aug_src='wordnet',name="wordnet_replace").augment

        char_param = {'aug_char_p': 0.1,'aug_word_p': 0.2,'include_upper_case': False,'include_numeric': False}
        self.typo = nac.KeyboardAug(**char_param, include_special_char=False,name="typo").augment
        self.swap_char = nac.RandomCharAug(**char_param, action="swap",name="char_swap").augment

        word_params = {"aug_p":0.1,"aug_max":10,}
        self.swap_word = naw.RandomWordAug(action="swap",**word_params,name="word_swap").augment
        self.del_word = naw.RandomWordAug(**word_params,name="word_delete").augment
        
        #if trnslt: self.back_translate=naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-zh',to_model_name='Helsinki-NLP/opus-mt-zh-en',device=_get_device(),name="backtranslate",).augment

        emb_param = {"top_k": 1,**word_params}
        if bert:
            self.context_insert = naw.ContextualWordEmbsAug(**emb_param,model_path='distilbert-base-uncased', action="insert",name="bert_insert").augment
            self.context_replacement = naw.ContextualWordEmbsAug(**emb_param,model_path='distilbert-base-uncased', action="substitute",name="bert_replace").augment
        
        if emb:
            p = _load_emb()
            self.glove_replace = naw.WordEmbsAug(**emb_param, model_path=p["glove"] ,model_type='glove',action="substitute",name="glove_replace").augment
            self.glove_insert = naw.WordEmbsAug(**emb_param, model_path=p["glove"], model_type='glove',action="insert",name="glove_insert").augment
            self.emb_replace = naw.WordEmbsAug(**emb_param, model_path=p["word2vec"] ,model_type='word2vec',action="substitute",name="w2v_replace").augment
            self.emb_insert = naw.WordEmbsAug(**emb_param, model_path=p["word2vec"], model_type='word2vec',action="insert",name="w2v_insert").augment


    def no_augmentations(self):
        return [self.unaugmented]


    def eda(self,K):
        if K not in (5,9,13,17):
            raise ValueError("EDA has 4 augmentations, so K should be 1+4X")
        transformation_list = [self.unaugmented,
                               self.synonym_wn, self.emb_insert, self.swap_word, self.del_word, #K=5
                               self.glove_replace, self.glove_insert,
                               self.swap_word, self.del_word, #K=9
                               self.emb_replace, self.emb_insert, self.swap_word, self.del_word, #K=13
                               self.context_replacement, self.context_insert, self.swap_word, self.del_word] #K=17
        return transformation_list[:K]
    

    def single_aug(self,aug):
        return [self.unaugmented,aug]
    
 # ----------------------------------------------------------------------------------------------------------------


def get_transforms_from_str(transform_name):
    if transform_name is None:
        return [lambda x: x]
    elif "precomputed" in transform_name:
        # of form eg: precomputed_2x3
        amounts = transform_name.split("_")[-1].split("x")
        precomp,aug = int(amounts[0]),int(amounts[1])
        if aug == 1: transform_list = [lambda x: x]
        elif aug == 2: 
            a = Augmentations(trnslt=False,bert=False,emb=False)
            transform_list = [a.unaugmented,a.swap_word]
        elif aug == 3:
            a = Augmentations(trnslt=False,bert=False,emb=False)
            transform_list = [a.unaugmented,a.synonym_wn,a.swap_word]
        else: raise NotImplementedError
        for i in range(precomp-1): 
            for j in range(aug):
                transform_list.append(None)
        assert len(transform_list) == precomp*aug
        return transform_list


    elif transform_name == "eda5":
        transform_list = Augmentations(trnslt=False,bert=False).eda(5)
    elif "eda" in transform_name:
        transform_list = Augmentations(trnslt=False).eda(int(transform_name[3:]))
    else:
        raise NotImplementedError # TODO add support for other augs, and maybe list constructor from string eg. "un_typ_syn_emb_del" -> [...]
    return transform_list


def _load_emb():
    _model_dir = os.environ.get("MODEL_DIR")

    if not os.path.exists(_model_dir):
        os.makedirs(_model_dir)

    w2vec_path = os.path.abspath(_model_dir+'GoogleNews-vectors-negative300.bin')
    if not os.path.isfile(w2vec_path):
        DownloadUtil.download_word2vec(dest_dir=_model_dir) # Download word2vec model

    glove_path = os.path.abspath(_model_dir+'glove.6B.300d.txt')
    if not os.path.isfile(glove_path):
        DownloadUtil.download_glove(model_name='glove.6B', dest_dir=_model_dir) # Download GloVe model

    #DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir=MODEL_DIR) # Download fasttext model
    paths = {"word2vec": w2vec_path,"glove":glove_path}
    return paths



def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"  


def main():
    os.environ["MODEL_DIR"] = "/home/spelz/AugMult_DP_NLP/augmentation_models"
    #sentence ="Sometimes to understand a word's meaning you need more than a definition; you need to see the word used in a sentence."
    sentence = """When you've got snow, it's really hard to learn a snow sport so we looked at all the different ways I could mimic being on snow without actually being on snow."""
    transform_list = get_transforms_from_str("eda17")
    for aug in transform_list:
        print(aug(sentence))

if __name__ == "__main__":
    main()