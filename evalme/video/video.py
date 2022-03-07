from collections import defaultdict

from label_studio_tools.postprocessing.video import extract_key_frames

from evalme.eval_item import EvalItem
from evalme.image.object_detection import ObjectDetectionEvalItem


class VideoEvalItem(EvalItem):
    SHAPE_KEY = 'videorectangle'

    def iou_over_time(self, pred, per_label=False):
        """
        IOU over time for video frames

        :param pred: Predicted VideoEvalItem
        :param per_label: per label calcu;ation flag
        :return: float or dict of floats
        """
        # prepare results vars for per_label
        if per_label:
            results = defaultdict(float)
            results_count = defaultdict(int)
        else:
            results = 0
            results_count = 0
        # extract ALL frames from raw data
        gt_frames_results = self.get_frames()
        pred_frames_results = pred.get_frames()
        # check each predicted frame set score with ground truth frame set
        for pred_frame in pred_frames_results:
            if per_label:
                max_value = {}
            else:
                max_value = 0.0
            for gt_frame in gt_frames_results:
                res = self.check_frames(gt_frames=gt_frame, pred_frames=pred_frame, per_label=per_label)
                if per_label:
                    if not max_value:
                        max_value = res
                    else:
                        max_value = max_value if list(res.values())[0] < list(max_value.values())[0] else res
                else:
                    max_value = max(max_value, res)
            if per_label:
                for key, value in max_value.items():
                    results[key] += value
                    results_count[key] += 1
            else:
                results += max_value
                results_count += 1
        # construct final scores
        if per_label:
            for key in results:
                results[key] = results[key] / results_count[key]
            return results
        return results / results_count if results_count else 0

    def get_frames(self):
        # extract frames from results
        if 'result' in self._raw_data:
            result = self._raw_data['result']
        else:
            result = self._raw_data
        return extract_key_frames(result)

    @staticmethod
    def check_frames(gt_frames, pred_frames, per_label=False):
        """
        Check frames score

        :param gt_frames: Ground truth frames
        :param pred_frames: Predicted frames
        :param per_label: Per_label calculation flag
        :return: score[float] or dict()
        """
        # check if labels
        labels_gt = gt_frames['value'].get('labels')
        labels_pred = pred_frames['value'].get('labels')
        if labels_gt != labels_pred:
            if per_label:
                return {label: 0 for label in labels_gt}
            return 0
        # extract frames from result
        gt_frames = gt_frames['value'].get('sequence', [])
        pred_frames = pred_frames['value'].get('sequence', [])
        if len(gt_frames) == 0 or len(pred_frames) == 0:
            if per_label:
                return {}
            return 0

        gt_frames = VideoEvalItem.transformed_frames(gt_frames)
        pred_frames = VideoEvalItem.transformed_frames(pred_frames)

        gt_keys = set(gt_frames.keys())
        pred_keys = set(pred_frames.keys())

        score = 0
        for i in gt_keys & pred_keys:
            t = ObjectDetectionEvalItem(raw_data=None)
            boxA = gt_frames.get(i)
            boxB = pred_frames.get(i)
            if boxA and boxB:
                score += t._iou(boxA=boxA, boxB=boxB)

        score = score / len(gt_keys | pred_keys)
        if per_label:
            return {label: score for label in labels_gt}
        return score

    @staticmethod
    def transformed_frames(frames):
        """
        Transform sequence to dict with frame as key
        :param frames: List of frames
        :return: Dict
        """
        result = {}
        for frame in frames:
            result[frame['frame']] = frame
        return result


def _as_video(item):
    if not isinstance(item, VideoEvalItem):
        return VideoEvalItem(item)
    return item


def video_iou(item_gt, item_pred, label_weights=None, per_label=False):
    item_gt = _as_video(item_gt)
    item_pred = _as_video(item_pred)
    return item_gt.iou_over_time(item_pred, per_label=per_label)
