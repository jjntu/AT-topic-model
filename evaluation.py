import json
import os
from copy import deepcopy
import itertools
import numpy as np
import pandas as pd

from typing import Mapping, Any, List, Tuple
from tqdm import tqdm

from bertopic import BERTopic
#from top2vec import Top2Vec

from cuml.manifold import UMAP
from octis.models.ETM import ETM
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from octis.models.CTM import CTM
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

class Trainer:
    def __init__(
        self,
        dataset: str,
        model_name: str,
        params: Mapping[str, Any],
        topk: int = 10,

        custom_dataset: bool = False,
        embeddings: np.ndarray = None,
        custom_model = None,
        verbose: bool = True,
        verbose_coherence: bool = False,
    ):
        
        self.dataset = dataset
        self.custom_dataset = custom_dataset
        self.model_name = model_name
        self.custom_model = custom_model
        self.params = params

        self.topk = topk
        self.embeddings = embeddings
        self.verbose = verbose
        self.verbose_coherence = verbose_coherence
        self.bertopic_model = None
        self.bertopics_topicseq = None

        
        self.metrics = None   
    
    def train(self, save: str = False) -> Mapping[str, Any]:
        results = []

        # Loop over all parameters
        params_name = list(self.params.keys())
        params = {
            param: (value if type(value) == list else [value])
            for param, value in self.params.items()
        }
        new_params = list(itertools.product(*params.values()))
        for param_combo in new_params:

            # Train and evaluate model
            params_to_use = {
                param: value for param, value in zip(params_name, param_combo) 
            }
            
            self.data = self.get_dataset()
            output = self._train_tm_model(params_to_use)
            self.metrics = self.get_metrics()
            scores = self.evaluate(output)
            
            if self.verbose_coherence:
                topic_npmi_zip = list(zip(scores["npmi_per_topic"], output["topics"]))
                del scores["npmi_per_topic"]

            # Update results
            result = {
                "Dataset": self.dataset,
                # "Dataset Size": len(self.data.get_corpus()),
                "Model": self.model_name,
                # "Params": params_to_use,
                "Scores": scores,
                "Topic Number": len(output["topics"]),
                "Topics": topic_npmi_zip if self.verbose_coherence else output["topics"],
            }
            results.append(result)

        if save:
            with open(f"{save}.json", "w") as f:
                json.dump(results, f, indent=4)

        return results
    
    def evaluate(self, output_tm):
        """Using metrics and output of the topic model, evaluate the topic model"""
        # Calculate results
        results = {}
        npmi_per_topic = []
        for scorers, _ in self.metrics:
            for scorer, name in scorers:                 
                score = scorer.score(output_tm)
                results[name] = float(score)

                if self.verbose_coherence and name == "npmi":
                    for topic in tqdm(output_tm["topics"]):
                        single_topic = {"topics": [topic]}
                        npmi_per_topic.append(float(scorer.score(single_topic)))
                    results["npmi_per_topic"] = npmi_per_topic
                        

        # Print results
        if self.verbose:
            print("Topics: ")
            for i, topic in enumerate(output_tm["topics"]):
                if i>20:
                    print("more ... ")
                    break
                joined_topic = " | ".join(topic)
                print(f"{i}: {joined_topic}")
            print("Score: ")
            for metric, score in results.items():
                if metric == "npmi_per_topic":
                    continue
                print(f"{metric}: {str(score)}")
            print("")

        return results
    
    def _train_tm_model(
        self, params: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], float]:
        """ Select and train the Topic Model """
        # Train BERTopic
        if self.model_name == "BERTopic":
            return self._train_bertopic(params)
        
        if self.model_name == "Top2Vec":
            return self._train_top2vec(params)

        # Train OCTIS model
        octis_models = ["ETM", "LDA", "CTM", "NMF"]
        if self.model_name in octis_models:
            return self._train_octis_model(params)
        
    def _train_bertopic(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train BERTopic model"""
        data = self.data.get_corpus()
        data = [" ".join(words) for words in data]
        params["calculate_probabilities"] = False

        if self.custom_model is not None:
            model = self.custom_model(**params)
        else:
            model = BERTopic(**params)

        topics, _ = model.fit_transform(data, self.embeddings)

        self.bertopic_model = model
        self.bertopics_topicseq = topics

        all_words = [word for words in self.data.get_corpus() for word in words]

        bertopic_topics = [
            [
                vals[0]
                vals[0] if vals[0] in all_words else all_words[0]
                for vals in model.get_topic(i)[:10]
            ]
            for i in range(len(set(topics)) - 1)
        ]
        
        output_tm = {"topics": bertopic_topics}

        return output_tm  
    
    def _train_octis_model(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train OCTIS model"""

        if self.model_name == "ETM":
            model = ETM(**params)
            model.use_partitions = False
        elif self.model_name == "LDA":
            model = LDA(**params)
            model.use_partitions = False
        elif self.model_name == "CTM":
            model = CTM(**params)
            model.use_partitions = False
        elif self.model_name == "NMF":
            model = NMF(**params)
            model.use_partitions = False

        output_tm = model.train_model(self.data)
        return output_tm

    def get_dataset(self):
        data = Dataset()
        
        if self.custom_dataset:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)

        return data
    
    def get_metrics(self):
        """Prepare evaluation measures using OCTIS"""
        texts = self.data.get_corpus()
        

        if self.bertopic_model is not None:
            docs = [" ".join(words) for words in texts]
            # Preprocess Documents
            documents = pd.DataFrame({"Document": docs,
                                    "ID": range(len(docs)),
                                    "Topic": self.bertopics_topicseq})
            documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
            cleaned_docs = self.bertopic_model._preprocess_text(documents_per_topic.Document.values)

            # Extract vectorizer and analyzer from BERTopic
            vectorizer = self.bertopic_model.vectorizer_model
            analyzer = vectorizer.build_analyzer()

            try:
                model = self.bertopic_model
                tokenizer = self.bertopic.tokenizer


                # summarize_topic_with_llm
                inputs = tokenizer.encode(self.data, return_tensors = 'pt', max_length=1024, truncation = True)
                summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # Extract features for Topic Coherence evaluation
                tokens = [analyzer(doc) for doc in cleaned_docs]
                summary.apply()
                texts = tokens

            except:
                tokens = [analyzer(doc) for doc in cleaned_docs]
                texts = tokens

            # Extract features for Topic Coherence evaluation
            tokens = [analyzer(doc) for doc in cleaned_docs]
            texts = tokens
        
        npmi = Coherence(texts=texts, topk=self.topk, measure="c_npmi")
        cv = Coherence(texts=texts, topk=self.topk, measure="c_v")
        
        topic_diversity = TopicDiversity(topk=self.topk)

        # Define methods
        coherence = [(npmi, "npmi"), (cv, "c_v")]
        diversity = [(topic_diversity, "diversity")]
        metrics = [(coherence, "Coherence"), (diversity, "Diversity")]

        return metrics