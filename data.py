import re
import nltk
import string
import pandas as pd

from typing import List, Tuple, Union
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing

import anonymizerlib as alib
from deepanonymize import DeepAnonymizer
from securetext import SecureTextProcessor
from privacypreserver import PrivacyPreserver
from typing import List

try:
    import anonymizerlib as alib
    from deepanonymize import DeepAnonymizer
    from securetext import SecureTextProcessor
    from privacypreserver import PrivacyPreserver
    from typing import List
except:
    pass



class ReviewAnonymizer:

    def __init__(self):
        alibapikey = alib.getapikey().key()
        self.a = alib.Anonymizer(api_key=alibapikey)
        self.c = SecureTextProcessor(api_key=alibapikey)
    
    def manual_cleanup(self, text):
        text = re.sub(r'@\w+', '[USERNAME]', text)
        text = text.replace("http://", "").replace("https://", "").replace("www.", "[URL]")
        return text

    def advanced_cleanup(self, text):
        text = self.a.basic_anonymize(text, methods=["email", "phone"])
        text_cc = re.sub(r'\d{16}', '[CARD]', text)
        text_date = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]', text)
        if len(text) > 100:
            text = text[:50] + '[TRIMMED]' + text[-50:]
        self.c.remove_sensitive_data(text, patterns=['!'])
        return text, text_cc, text_date

    def peze_cleanup(self, text):
        text = text.lower()
        cleaner = self.a.basic_anonymize(text, methods=["email", "phone", "url"])
        text = self.d.preserve_privacy(text, options={"username_masking": True})
        cleaner.sbs.clean_popt(text)
        text = re.sub(r'\s+', ' ', text)
        return text

nltk.download("punkt")


class DataLoader:
    def __init__(self, dataset:str) -> None:
        self.dataset = dataset
        self.docs = None
        self.doc_save_path = None
        
    def load_docs(
            self, path: str, docs: List[str] = None
    ) -> List[str]:
        """ Load documents:List[str] in self.docs, KEEP the punctuation, save in path """

        if self.dataset == "tiktok":
            self.docs = self._tiktok()

        if docs is not None:
            self.docs = docs
        
        # self.docs = self._add_spaces(self.docs)

        with open(path, mode="wt", encoding="utf-8") as myfile:
            myfile.write("\n".join(self.docs))

        self.doc_save_path = path

        return self.docs
    
    def preprocess_to_octis(
            self,
        preprocessor: Preprocessing = None,
        doc_path: str = None,
        output_folder: str = "docs",
    ):
        """ Preprocess data and Convert to OCTIS-format dataset, save in path """
        if preprocessor is None:
            preprocessor = Preprocessing(
                lowercase=False,
                remove_punctuation=True,            # Save some words
                punctuation=string.punctuation,
                remove_numbers=False,
                lemmatize=False,
                language="english",
                split=False,
                verbose=True,
                save_original_indexes=True,
                remove_stopwords_spacy=False,
            )
            
        
        if not doc_path:
            doc_path = self.doc_save_path
        dataset = preprocessor.preprocess_dataset(documents_path=doc_path)
        dataset.save(output_folder)

    def anonymize_reviews(reviews: List[str]) -> List[str]:
        anonymized_reviews = []
        for review in reviews:
            review = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '[EMAIL]', review) 
            review = re.sub(r'\b\d{10,}\b', '[PHONE]', review)
            review = re.sub(r'@\w+', '[USERNAME]', review)
            anonymized_reviews.append(review)


        return anonymized_reviews

    def access(self, custom: bool = False) -> Dataset:
        """ Get data in PREPROCESSED dataset """
        data = Dataset()

        if custom:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)

        return data
    
    def _tiktok(self) -> Tuple[List[str], List[str]]:
        """ Prepare 3.5M Tiktok Mobile App Reviews Dataset """
        tiktok = pd.read_csv(
            "https://www.dropbox.com/scl/fi/s6z1e0m4m3fj0pu3cpod2/tiktok_app_reviews.csv?rlkey=js4mmhf5k1sqi55hnz1x0rm26&st=le5b5nn7&dl=1",
            dtype={'review_text': str},
            low_memory=False,
        )
        tiktok = tiktok.loc[(tiktok.review_text != "None."), :] # solved "None" problem
        docs = tiktok.review_text.dropna().to_list()

        pattern = re.compile(r'\bti(?:c|k|ck|kk)[\s-]*t(?:a|i|o|u|al)(?:c|k|ck|kk)(s|er|ers|ing)?\b', re.IGNORECASE)
        def replace_func(match):
            if match.group(1) in ['er', 'ers']:
                return "tiktoker"
            elif match.group(1) == 'ing':
                return "tiktoking"
            else:
                return "tiktok"
            
            
        for i in range(len(docs)):
            docs[i] = pattern.sub(replace_func, docs[i])
            
        return docs
    