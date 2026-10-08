# Experiment framework
import operator
import random
import math
import heapq 

from collections import defaultdict, Counter
from typing import Any, Callable, Optional

import numpy as np
from compressors import DefaultCompressor
from tqdm import tqdm


class KnnExpText:
    def __init__(
        self,
        aggregation_function: Callable,
        compressor: DefaultCompressor,
        distance_function: Callable,
        data: list
    ) -> None:
        self.aggregation_func = aggregation_function
        self.compressor = compressor
        self.distance_func = distance_function
        self.distance_matrix: list = []
        self.document_count = len(data)
        self.inverted_index = defaultdict(list)
        self.euclidean_lengths = defaultdict(float)
        self.create_inverted_index(data)
        self.create_euclidean_length_index(data)

    def create_inverted_index(self, data: list) -> None:
        """
        Constructs an inverted index mapping terms in `data` to the indices
        where they appear. Ensure that `data` has already been preprocessed.

        term -> posting list
        
        A posting list contains tuples of (docID, term_frequency).

        Arguments:
            data (list[str]): Corpus upon which index will be constructed.

        Returns:
            None: None
        """

        for index, row in enumerate(data):
            tokens = Counter(row.split())

            for token, frequency in tokens.items():
                self.inverted_index[token].append((index, frequency))

    def create_euclidean_length_index(self, data: list) -> None:
        """
        Constructs an index mapping indices in `data` to the Euclidean lengths
        of their vectors (tf-idf weights).

        index -> Euclidean length of `data[index]`

        Arguments:
            data (list[str]): Corpus upon which index will be constructed.

        Returns:
            None: None
        """

        for index, row in enumerate(data):
            tokens = Counter(row.split())

            normalization_factors = []
            for token, frequency in tokens.items():
                factor = self.compute_tf_idf(token, frequency)
                normalization_factors.append(factor)

            self.euclidean_lengths[index] = self.compute_euclidean_length(normalization_factors)

    def calc_dis(
        self, data: list, train_data: Optional[list] = None, fast: bool = False
    ) -> None:
        """
        Calculates the distance between either `data` and itself or `data` and
        `train_data` and appends the distance to `self.distance_matrix`.

        Arguments:
            data (list): Data to compute distance between.
            train_data (list): [Optional] Training data to compute distance from `data`.
            fast (bool): [Optional] Uses the _fast compression length function
                                    of `self.compressor`.

        Returns:
            None: None
        """

        data_to_compare = data
        if train_data is not None:
            data_to_compare = train_data

        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            if fast:
                t1_compressed = self.compressor.get_compressed_len_fast(t1)
            else:
                t1_compressed = self.compressor.get_compressed_len(t1)
            for j, t2 in enumerate(data_to_compare):
                if fast:
                    t2_compressed = self.compressor.get_compressed_len_fast(t2)
                    t1t2_compressed = self.compressor.get_compressed_len_fast(
                        self.aggregation_func(t1, t2)
                    )
                else:
                    t2_compressed = self.compressor.get_compressed_len(t2)
                    t1t2_compressed = self.compressor.get_compressed_len(
                        self.aggregation_func(t1, t2)
                    )
                distance = self.distance_func(
                    t1_compressed, t2_compressed, t1t2_compressed
                )
                distance4i.append(distance)
            self.distance_matrix.append(distance4i)

    def calc_dis_with_single_compressed_given(
        self, data: list, data_len: list = None, train_data: Optional[list] = None
    ) -> None:
        """
        Calculates the distance between either `data`, `data_len`, or
        `train_data` and appends the distance to `self.distance_matrix`.

        Arguments:
            data (list): Data to compute distance between.
            train_data (list): [Optional] Training data to compute distance from `data`.
            fast (bool): [Optional] Uses the _fast compression length function
                                    of `self.compressor`.

        Returns:
            None: None
        """

        data_to_compare = data
        if train_data is not None:
            data_to_compare = train_data

        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            t1_compressed = self.compressor.get_compressed_len_given_prob(
                t1, data_len[i]
            )
            for j, t2 in tqdm(enumerate(data_to_compare)):
                t2_compressed = self.compressor.get_compressed_len_given_prob(
                    t2, data_len[j]
                )
                t1t2_compressed = self.compressor.get_compressed_len(
                    self.aggregation_func(t1, t2)
                )
                distance = self.distance_func(
                    t1_compressed, t2_compressed, t1t2_compressed
                )
                distance4i.append(distance)
            self.distance_matrix.append(distance4i)

    def calc_dis_single(self, t1: str, t2: str) -> float:
        """
        Calculates the distance between `t1` and `t2` and returns
        that distance value as a float-like object.

        Arguments:
            t1 (str): Data 1.
            t2 (str): Data 2.

        Returns:
            float-like: Distance between `t1` and `t2`.
        """

        t1_compressed = self.compressor.get_compressed_len(t1)
        t2_compressed = self.compressor.get_compressed_len(t2)
        t1t2_compressed = self.compressor.get_compressed_len(
            self.aggregation_func(t1, t2)
        )
        distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
        return distance

    def calc_dis_single_multi(self, train_data: list, datum: str) -> list:
        """
        Calculates the distance between `train_data` and `datum` and returns
        that distance value as a float-like object.

        Arguments:
            train_data (list): Training data as a list-like object.
            datum (str): Data to compare against `train_data`.

        Returns:
            list: Distance between `t1` and `t2`.
        """

        distance4i = []
        t1_compressed = self.compressor.get_compressed_len(datum)
        for j, t2 in tqdm(enumerate(train_data)):
            t2_compressed = self.compressor.get_compressed_len(t2[1])
            t1t2_compressed = self.compressor.get_compressed_len(
                self.aggregation_func(datum, t2[1])
            )
            distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
            distance4i.append(distance)
        return distance4i

    def calc_dis_with_vector(self, data: list, train_data: Optional[list] = None):
        """
        Calculates the distance between `train_data` and `data` and returns
        that distance value as a float-like object.

        Arguments:
            train_data (list): Training data as a list-like object.
            datum (str): Data to compare against `train_data`.

        Returns:
            float-like: Distance between `t1` and `t2`.
        """

        if train_data is not None:
            data_to_compare = train_data
        else:
            data_to_compare = data
        for i, t1 in tqdm(enumerate(data)):
            distance4i = []
            for j, t2 in enumerate(data_to_compare):
                distance = self.distance_func(t1, t2)
                distance4i.append(distance)
            self.distance_matrix.append(distance4i)

    def calc_acc(
        self,
        k: int,
        label: list,
        train_label: Optional[list] = None,
        provided_distance_matrix: Optional[list] = None,
        rand: bool = False,
    ) -> tuple:
        """
        Calculates the accuracy of the algorithm.

        Arguments:
            k (int?): TODO
            label (list): Predicted Labels.
            train_label (list): Correct Labels.
            provided_distance_matrix (list): Calculated Distance Matrix to use
                                             instead of `self.distance_matrix`.
            rand (bool): TODO

        Returns:
            tuple: predictions, and list of bools indicating prediction correctness.

        """
        if provided_distance_matrix is not None:
            self.distance_matrix = provided_distance_matrix
        correct = []
        pred = []
        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k + 1

        for i in range(len(self.distance_matrix)):
            sorted_idx = np.argpartition(np.array(self.distance_matrix[i]), range(k))
            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1
            sorted_pred_lab = sorted(
                pred_labels.items(), key=operator.itemgetter(1), reverse=True
            )
            most_count = sorted_pred_lab[0][1]
            most_label = sorted_pred_lab[0][0]

            # kNN classifier based on a plurality vote. If there exists a counting
            # tie (e.g. 1-1-1 for k=3), it considers the closest neighbor based on NCD.
            # If there are ties in NCD, it considers which label appears first in the
            # training data.
            if_right = 0
            if most_label == label[i]:
                if_right = 1

            pred.append(most_label)
            correct.append(if_right)
        print("Accuracy is {}".format(sum(correct) / len(correct)))
        return pred, correct

    def combine_dis_acc(
        self,
        k: int,
        data: list,
        label: list,
        train_data: Optional[list] = None,
        train_label: Optional[list] = None,
    ) -> tuple:
        """
        Calculates the distance and the accuracy of the algorithm for data with
        training.

        Arguments:
            k (int?): TODO
            data (list): Data used for predictions.
            label (list): Predicted Labels.
            train_data (list): Training data to compare distances.
            train_label (list): Correct Labels.

        Returns:
            tuple: predictions, and list of bools indicating prediction correctness.
        """

        correct = []
        pred = []
        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k + 1
        if train_data is not None:
            data_to_compare = train_data
        else:
            data_to_compare = data
        for i, t1 in tqdm(enumerate(data)):
            distance4i = self.calc_dis_single_multi(data_to_compare, t1)
            sorted_idx = np.argsort(np.array(distance4i))
            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1
            sorted_pred_lab = sorted(
                pred_labels.items(), key=operator.itemgetter(1), reverse=True
            )
            most_count = sorted_pred_lab[0][1]
            if_right = 0
            most_label = sorted_pred_lab[0][0]
            for pair in sorted_pred_lab:
                if pair[1] < most_count:
                    break
                if pair[0] == label[i]:
                    if_right = 1
                    most_label = pair[0]
            pred.append(most_label)
            correct.append(if_right)
        print("Accuracy is {}".format(sum(correct) / len(correct)))
        return pred, correct

    def filter_candidates(self, train_data: list, datum: str):
        """
        Filters `train_data` to only the documents relevant to `datum`. 

        Arguments:
            train_data (list[str]): Training data to be filtered.
            datum (str): Search string.

        Returns:
            candidate_data (list[(int, str)]): Filtered candidates that
                                              show relevance to `datum`.
        """

        candidate_data = []

        top_k_scores = self.compute_document_scores(train_data, datum, 100)
        for index in top_k_scores:
            candidate_data.append((index, train_data[index]))

        relevant_document_count = len(top_k_scores)
        return candidate_data, relevant_document_count

    def compute_document_scores(self, train_data: list, datum: str, top_k: int) -> list:
        """
        Documents in `train_data` are scored and ranked based on cosine 
        similarity to `datum`. Vectors make use of tf-idf weights.

        Arguments:
            train_data (list[str]): Training data to be scored and ranked.
            datum (str): Search string.
            top_k (int): Number of documents to be returned.

        Returns:
            (list): Indices of top_k documents.
        """
        
        datum_tokens = Counter(datum.split())

        scores = defaultdict(float)

        for token, token_freq in datum_tokens.items():
            posting_list = self.inverted_index.get(token)
            if not posting_list:
                continue

            idf = math.log(self.document_count/len(posting_list))
            datum_tf_idf = token_freq * idf
            for doc_id, term_freq in self.inverted_index[token]:
                doc_tf_idf = term_freq * idf

                scores[doc_id] += datum_tf_idf * doc_tf_idf

        for doc_id, score in scores.items():
            scores[doc_id] = scores[doc_id] / self.euclidean_lengths[doc_id]

        return heapq.nlargest(top_k, scores, key=scores.get)

    def compute_euclidean_length(self, factors: list) -> float:
        """
        Computes the Euclidean length of a vector based on 
        the factors (components) in the vector.

        Arguments:
            factors (list): Components of the vector.

        Returns:
            result (float): Euclidean length of vector.
        """

        result = math.sqrt(sum(factor ** 2 for factor in factors))
        return result

    def compute_tf_idf(self, term: str, term_freq: int) -> float:
        """
        Computes the Term Frequency-Inverse Document Frequency (TF-IDF)
        weight of the given term.

        Arguments:
            term (str): Term to be considered.
            term_freq (int): Frequency of term in document

        Returns:
            tf_idf (float): TF-IDF weight.
        """
        posting_list = self.inverted_index[term]
        doc_freq = len(posting_list)
        idf = math.log(self.document_count/doc_freq)
        tf_idf = term_freq * idf
        return tf_idf

    def combine_dis_acc_single(
        self,
        k: int,
        train_data: list,
        train_label: list,
        datum: str,
        label: Any,  # int, as used in this application
    ) -> tuple:
        """
        Calculates the distance and the accuracy of the algorithm for a single
        datum with training.

        Arguments:
            k (int?): TODO
            train_data (list): Training data to compare distances.
            train_label (list): Correct Labels.
            datum (str): Datum used for predictions.
            label (Any): Correct label of datum.

        Returns:
            tuple: prediction, and a bool indicating prediction correctness.
        """
        
        candidate_data, hits = self.filter_candidates(train_data, datum)
        # Support multi processing - must provide train data and train label
        distance4i = self.calc_dis_single_multi(candidate_data, datum)
        sorted_idx = np.argpartition(np.array(distance4i), range(k))
        sorted_indices = [candidate_data[idx][0] for idx in sorted_idx]
        pred_labels = defaultdict(int)
        for j in range(k):
            pred_l = train_label[sorted_indices[j]]
            pred_labels[pred_l] += 1
        sorted_pred_lab = sorted(
            pred_labels.items(), key=operator.itemgetter(1), reverse=True
        )
        most_count = sorted_pred_lab[0][1]
        most_label = sorted_pred_lab[0][0]
        
        # kNN classifier based on a plurality vote. If there exists a counting
        # tie (e.g. 1-1-1 for k=3), it considers the closest neighbor based on NCD.
        # If there are ties in NCD, it considers which label appears first in the
        # training data.
        if_right = 0
        if most_label == label:
            if_right = 1

        pred = most_label
        correct = if_right
        return pred, correct, hits
