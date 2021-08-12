from typing import Dict, Any
import logging
import numpy as np
from scipy import stats
import torch
from torch_geometric.nn.unpool import knn_interpolate

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


class S3DISTracker(SegmentationTracker):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._test_area = None
        self._full_vote_miou = None
        self._vote_miou = None
        self._full_confusion = None
        self._iou_per_class = {}

    def track(self, model: model_interface.TrackerInterface, full_res=False, data=None, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        # Train mode or low res, nothing special to do
        if self._stage == "train" or not full_res:
            return

        # Test mode, compute votes in order to get full res predictions
        if self._test_area is None:
            self._test_area = self._dataset.test_data.clone()
            if self._test_area.y is None:
                raise ValueError("It seems that the test area data does not have labels (attribute y).")
            self._test_area.prediction_count = torch.zeros(self._test_area.y.shape[0], dtype=torch.int)
            self._test_area.votes = torch.zeros((self._test_area.y.shape[0], self._num_classes), dtype=torch.float)
            self._test_area.to(model.device)

        # Gather origin ids and check that it fits with the test set
        inputs = data if data is not None else model.get_input()
        if inputs[SaveOriginalPosId.KEY] is None:
            raise ValueError("The inputs given to the model do not have a %s attribute." % SaveOriginalPosId.KEY)

        originids = inputs[SaveOriginalPosId.KEY]
        if originids.dim() == 2:
            originids = originids.flatten()
        if originids.max() >= self._test_area.pos.shape[0]:
            raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

        # Set predictions
        outputs = model.get_output()
        self._test_area.votes[originids] += outputs
        self._test_area.prediction_count[originids] += 1

    def finalise(self, full_res=False, vote_miou=True, ply_output="", **kwargs):
        per_class_iou = self._confusion_matrix.get_intersection_union_per_class()[0]
        self._iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: v for k, v in enumerate(per_class_iou)}

        if vote_miou and self._test_area:
            # Complete for points that have a prediction
            self._test_area = self._test_area.to("cpu")
            c = ConfusionMatrix(self._num_classes)
            has_prediction = self._test_area.prediction_count > 0
            gt = self._test_area.y[has_prediction].numpy()
            pred = torch.argmax(self._test_area.votes[has_prediction], 1).numpy()
            c.count_predicted_batch(gt, pred)
            self._vote_miou = c.get_average_intersection_union() * 100

        if full_res:
            self._compute_full_miou()

        if ply_output:
            has_prediction = self._test_area.prediction_count > 0
            self._dataset.to_ply(
                self._test_area.pos[has_prediction].cpu(),
                torch.argmax(self._test_area.votes[has_prediction], 1).cpu().numpy(),
                ply_output,
            )

    def _compute_full_miou(self):
        if self._full_vote_miou is not None:
            return

        has_prediction = self._test_area.prediction_count > 0
        log.info(
            "Computing full res mIoU, we have predictions for %.2f%% of the points."
            % (torch.sum(has_prediction) / (1.0 * has_prediction.shape[0]) * 100)
        )

        self._test_area = self._test_area.to("cpu")

        # Full res interpolation
        full_pred = knn_interpolate(
            self._test_area.votes[has_prediction], self._test_area.pos[has_prediction], self._test_area.pos, k=1,
        )

        # Full res pred
        self._full_confusion = ConfusionMatrix(self._num_classes)
        self._full_confusion.count_predicted_batch(self._test_area.y.numpy(), torch.argmax(full_pred, 1).numpy())
        self._full_vote_miou = self._full_confusion.get_average_intersection_union() * 100

    @property
    def full_confusion_matrix(self):
        return self._full_confusion

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        if verbose:
            metrics["{}_iou_per_class".format(self._stage)] = self._iou_per_class
            if self._vote_miou:
                metrics["{}_full_vote_miou".format(self._stage)] = self._full_vote_miou
                metrics["{}_vote_miou".format(self._stage)] = self._vote_miou
        return metrics

class S3DISInsTracker(SegmentationTracker):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._mucov = None
        self._mwcov = None
        self._precision = None
        self._recall = None
        self.tpsins = [[] for itmp in range(self._num_classes)]
        self.fpsins = [[] for itmp in range(self._num_classes)]
        self.all_mean_cov = [[] for itmp in range(self._num_classes)]
        self.all_mean_weighted_cov = [[] for itmp in range(self._num_classes)]

    def track(self, model: model_interface.TrackerInterface, full_res=False, **kwargs):
        super().track(model)

        # Train mode or low res, nothing special to do
        if self._stage == "train" or not full_res:
            return

        ins_outputs = model.get_ins_output()
        ins_targets = model.get_int_labels()
        sem_outputs = model.get_output()
        sem_targets = model.get_labels()
        self._ins_compute_metrics(sem_outputs, ins_outputs, sem_targets, ins_targets)

    def _ins_compute_metrics(self, sem_outputs, ins_outputs, sem_gt, ins_gt):
        mask = ins_gt != self._ignore_label
        sem_outputs = sem_outputs[mask]
        ins_outputs = ins_outputs[mask]
        sem_gt = sem_gt[mask]
        ins_gt = ins_gt[mask]

        total_gt_ins = np.zeros(self._num_classes)
        at = 0.5

        un = np.unique(ins_outputs)
        pts_in_pred = [[] for itmp in range(self._num_classes)]
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = (sem_outputs == g)
            sem_seg_i = int(stats.mode(sem_outputs[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]

        un = np.unique(ins_gt)
        pts_in_gt = [[] for itmp in range(self._num_classes)]
        for ig, g in enumerate(un):
            tmp = (ins_gt == g)
            sem_seg_i = int(stats.mode(sem_gt[tmp])[0])
            pts_in_gt[sem_seg_i] += [tmp]

        # instance mucov & mwcov
        for i_sem in range(self._num_classes):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                ovmax = 0.
                num_ins_gt_point = np.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        ipmax = ip

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                self._all_mean_cov[i_sem].append(mean_cov)

                mean_weighted_cov /= num_gt_point
                self._all_mean_weighted_cov[i_sem].append(mean_weighted_cov)


        # instance precision & recall
        for i_sem in range(self._num_classes):
            tp = [0.] * len(pts_in_pred[i_sem])
            fp = [0.] * len(pts_in_pred[i_sem])
            gtflag = np.zeros(len(pts_in_gt[i_sem]))
            total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                ovmax = -1.

                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    union = (ins_pred | ins_gt)
                    intersect = (ins_pred & ins_gt)
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        igmax = ig

                if ovmax >= at:
                        tp[ip] = 1  # true
                else:
                    fp[ip] = 1  # false positive

            self._tpsins[i_sem] += tp
            self._fpsins[i_sem] += fp

    def finalise(self, full_res=False, **kwargs):
        if full_res:
            mucov = np.zeros(self._num_classe)
            mwcov = np.zeros(self._num_classe)
            for i_sem in range(self._num_classes):
                mucov[i_sem] = np.mean(self._all_mean_cov[i_sem])
                mwcov[i_sem] = np.mean(self._all_mean_weighted_cov[i_sem])
            self._mucov = np.mean(mucov)
            self._mwcov = np.mean(mwcov)

            precision = np.zeros(self._num_classes)
            recall = np.zeros(self._num_classes)
            for i_sem in range(self._num_classes):
                tp = np.asarray(self._tpsins[i_sem]).astype(np.float)
                fp = np.asarray(self._fpsins[i_sem]).astype(np.float)
                tp = np.sum(tp)
                fp = np.sum(fp)
                rec = tp / self._total_gt_ins[i_sem]
                prec = tp / (tp + fp)

                precision[i_sem] = prec
                recall[i_sem] = rec

            self._precision = np.mean(precision)
            self._recall = np.mean(recall)

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        metrics = super().get_metrics(verbose)
        if verbose:
            if self._precision is not None:
                # instance segmentation
                metrics["{}_precision".format(self._stage)] = self._precision
                metrics["{}_recall".format(self._stage)] = self._recall
                metrics["{}_mwcov".format(self._stage)] = self._mwcov
                metrics["{}_mucov".format(self._stage)] = self._mucov
        return metrics

