import numpy as np
import csv
import os
from configs import PreprocessingConfig
from dataclasses import fields

class CandidatesReport:
    def __init__(
        self,
        directory: str,
        dataset: str,
        train_rows: int,
        test_rows: int,
        candidate_count: int,
        examine_candidates: int,
        *,
        mean: float,
    ):
        self.directory = directory
        self.dataset = dataset
        self.train_rows = train_rows
        self.test_rows = test_rows
        self.candidate_count = candidate_count
        self.examine_candidates = examine_candidates
        self.mean = mean
        self.create_candidates_report()

    def create_candidates_report(self):
        self._print_candidates_report()
        self._save_candidates_report()

    def _get_report_rows(self):
        if self.examine_candidates == 0:
            mean_value = "cos sim accuracy"
        else:
            mean_value = "intersection ratio"

        return [
            ["dataset", self.dataset],
            ["train rows", self.train_rows],
            ["test rows", self.test_rows],
            ["candidate count", self.candidate_count],
            [f"average {mean_value}", self.mean],
        ]

    def _save_candidates_report(self):
        if self.directory is None:
            return
        
        if self.examine_candidates == 0:
            mean_value = "accuracy"
        else:
            mean_value = "ratio"
        
        path = os.path.join(
            self.directory,
            f"{self.dataset}_{self.candidate_count}_candidates_{mean_value}.csv"
        )

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self._get_report_rows())

    def _print_candidates_report(self):
        for label, value in self._get_report_rows():
            print(f"{label}: {value}")


class GenerateReport:
    def __init__(
        self, 
        labels: list,
        predictions: list,
        times: list,
        dataset: str,
        directory: str,
        k: int,
        candidate_count: int,
        preprocessing_config: PreprocessingConfig
    ):
        self.labels = labels
        self.predictions = predictions
        self.times = times
        self.dataset = dataset
        self.directory = directory
        self.k = k
        self.candidate_count = candidate_count
        self.preprocessing_config = preprocessing_config
        self.classes = sorted(set(self.labels) | set(self.predictions))
        self.create_metrics_report()
        
        
    def create_metrics_report(self):
        matrix = self.create_confusion_matrix()

        col_sums = matrix.sum(axis=0)  
        row_sums = matrix.sum(axis=1)
        true_pos_vec = np.diag(matrix)

        accuracy = true_pos_vec.sum() / row_sums.sum()
        precision_vec = self._safe_divide(true_pos_vec, col_sums)
        recall_vec = self._safe_divide(true_pos_vec, row_sums)
        f1_vec = self._safe_divide(2 * precision_vec * recall_vec, precision_vec + recall_vec)

        times = np.array(self.times)
        runs = len(times)
        pre_avg_time = times[:,0].mean()
        index_avg_time = times[:,1].mean()
        exp_avg_time = times[:,2].mean()
        total_avg_time = pre_avg_time + index_avg_time + exp_avg_time
        avg_times = [pre_avg_time, index_avg_time, exp_avg_time, total_avg_time]

        report_args = (precision_vec, recall_vec, f1_vec, accuracy, avg_times, runs)
        self._print_report(*report_args)
        if self.directory is not None:
            self._save_report_csv(*report_args)
        

    def create_confusion_matrix(self):
        if len(self.labels) != len(self.predictions):
            raise ValueError("Labels and predictions must have the same length.")

        label_to_index = {label: idx for idx, label in enumerate(self.classes)}
        class_count = len(self.classes)
        
        confusion_matrix = np.zeros((class_count, class_count), dtype=int)
        for lab, pred in zip(self.labels, self.predictions):
            confusion_matrix[label_to_index[lab]][label_to_index[pred]] += 1

        return confusion_matrix   
    

    def _safe_divide(
        self,
        num: np.ndarray, 
        den: np.ndarray
    ):
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)


    def _build_report(self, precision_vec, recall_vec, f1_vec, accuracy, times, runs):
        configs = [
            field.name
            for field in fields(self.preprocessing_config)
            if getattr(self.preprocessing_config, field.name)
        ]

        class_rows = [
            (str(cls), p, r, f)
            for cls, p, r, f in zip(self.classes, precision_vec, recall_vec, f1_vec)
        ]
        macro_row = ("macro avg", precision_vec.mean(), recall_vec.mean(), f1_vec.mean())

        summary_rows = [
            ("accuracy", accuracy),
            ("macro avg", macro_row[1:]),
            ("preprocessing avg time", times[0]),
            ("index creation avg time", times[1]),
            ("experiment avg time", times[2]),
            ("total avg time", times[3]),
            ("runs", runs),
            ("k", self.k),
            ("candidate count", self.candidate_count),
            ("preprocessing", ", ".join(configs)),
            ("dataset", self.dataset),
        ]
        return class_rows, summary_rows
        

    @staticmethod
    def _fmt(value):
        return f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value)
        

    def _print_report(self, *report_args):
        class_rows, summary_rows = self._build_report(*report_args)

        labels = [r[0] for r in class_rows] + [r[0] for r in summary_rows]
        w = max(len(l) for l in labels)

        print(f"{'':>{w}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}\n")

        for label, *vals in class_rows:
            print(f"{label:>{w}}  " + "  ".join(f"{self._fmt(v):>9}" for v in vals))
        print()

        for label, value in summary_rows:
            if isinstance(value, tuple):
                cells = [self._fmt(v) for v in value]
            else:
                cells = ["", "", self._fmt(value)]
            print(f"{label:>{w}}  " + "  ".join(f"{c:>9}" for c in cells))


    def _save_report_csv(self, *report_args):
        class_rows, summary_rows = self._build_report(*report_args)

        path = os.path.join(self.directory, self.dataset)
        with open(f"{path}_{self.candidate_count}_{self.k}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "precision", "recall", "f1-score"])

            for label, *vals in class_rows:
                writer.writerow([label, *map(self._fmt, vals)])

            for label, value in summary_rows:
                vals = value if isinstance(value, tuple) else (value,)
                writer.writerow([label, *map(self._fmt, vals)])

                
if __name__ == "__main__":
    pass