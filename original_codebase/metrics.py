import numpy as np
import csv
import os

class GenerateReport:
    def __init__(
        self, 
        labels: list,
        predictions: list,
        times: list,
        dataset: str,
        directory: str,
        k: int
    ):
        self.labels = labels
        self.predictions = predictions
        self.times = times
        self.dataset = dataset
        self.directory = directory
        self.k = k
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
        exp_avg_time = times[:,1].mean()
        total_avg_time = pre_avg_time + exp_avg_time
        avg_times = [pre_avg_time, exp_avg_time, total_avg_time]

        self._print_report(
            precision_vec,
            recall_vec,
            f1_vec,
            accuracy,
            avg_times,
            runs,
        )
        if self.directory is not None:
            self._save_report_csv(
                precision_vec,
                recall_vec,
                f1_vec,
                accuracy,
                avg_times,
                runs,
            )
        

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


    def _print_report(
        self,
        precision_vec: np.ndarray,
        recall_vec: np.ndarray,
        f1_vec: np.ndarray,
        accuracy,
        times: list,
        runs: int,
    ):
        name_width = max(len(str(c)) for c in self.classes + ["preprocessing avg time"])
        header = f"{'':>{name_width}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}"
        print(header)
        print()

        for cls, p, r, f in zip(self.classes, precision_vec, recall_vec, f1_vec):
            print(f"{str(cls):>{name_width}}  {p:>9.5f}  {r:>9.5f}  {f:>9.5f}")
        print()

        print(f"{'accuracy':>{name_width}}  {'':>9}  {'':>9}  {accuracy:>9.5f}")
        print(
            f"{'macro avg':>{name_width}}  "
            f"{precision_vec.mean():>9.5f}  {recall_vec.mean():>9.5f}  {f1_vec.mean():>9.5f}"
        )
        print(f"{'preprocessing avg time':>{name_width}}  {'':>9}  {'':>9}  {times[0]:>9.5f}")
        print(f"{'experiment avg time':>{name_width}}  {'':>9}  {'':>9}  {times[1]:>9.5f}")
        print(f"{'total avg time':>{name_width}}  {'':>9}  {'':>9}  {times[2]:>9.5f}")
        print(f"{'runs':>{name_width}}  {'':>9}  {'':>9}  {runs:>9}")
        print(f"{'k':>{name_width}}  {'':>9}  {'':>9}  {self.k:>9}")
        print(f"{'dataset':>{name_width}}  {'':>9}  {'':>9}  {self.dataset:>9}")


    def _save_report_csv(
        self,
        precision_vec: np.ndarray,
        recall_vec: np.ndarray,
        f1_vec: np.ndarray,
        accuracy,
        times: list,
        runs: int,
    ):
        path = os.path.join(self.directory, self.dataset)
        with open(f"{path}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "precision", "recall", "f1-score"])

            for cls, p, r, f1 in zip(self.classes, precision_vec, recall_vec, f1_vec):
                writer.writerow([cls, f"{p:.5f}", f"{r:.5f}", f"{f1:.5f}"])

            writer.writerow(["accuracy", f"{accuracy:.5f}"])
            writer.writerow([
                "macro avg",
                f"{precision_vec.mean():.5f}",
                f"{recall_vec.mean():.5f}",
                f"{f1_vec.mean():.5f}",
            ])
            writer.writerow(["preprocessing avg time", f"{times[0]:.5f}"])
            writer.writerow(["experiment avg time", f"{times[1]:.5f}"])
            writer.writerow(["total avg time", f"{times[2]:.5f}"])
            writer.writerow(["runs", runs])
            writer.writerow(["k", self.k])
            writer.writerow(["dataset", self.dataset])

if __name__ == "__main__":
    pass