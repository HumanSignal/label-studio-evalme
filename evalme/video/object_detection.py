from copy import deepcopy

from evalme.eval_item import EvalItem

class VideoObjectDetectionEvalItem(EvalItem):
    SHAPE_KEY = 'videorectanglelabels'

    def extract_key_frames(self):
        """
        Extract frames from key frames
        :return: List of frames
        """
        final_results = []
        for result in self._raw_data:
            sequence = result['value']['sequence']
            if len(sequence) < 1:
                continue
            label = result['value'].get('labels', "")
            sequence = sorted(sequence, key=lambda d: d['frame'])
            if len(sequence) < 2:
                element = sequence.pop()
                final_results.extend(
                    self._construct_result_from_frames(frame1=element,
                                                       frame2={},
                                                       res_type="rectanglelabels",
                                                       res=result,
                                                       label=label,
                                                       frameCount=result["value"].get("frameCount", 0),
                                                       exclude_first=False)
                )
            else:
                exclude_first = False
                for i in range(len(sequence)):
                    frame_a = sequence[i]
                    frame_b = {} if i == len(sequence)-1 else sequence[i+1]
                    final_results.extend(self._construct_result_from_frames(frame1=frame_a,
                                                                            frame2=frame_b,
                                                                            res_type="rectanglelabels",
                                                                            res=result,
                                                                            label=label,
                                                                            frameCount=result["value"].get("frameCount", 0),
                                                                            exclude_first=exclude_first))
                    exclude_first = frame_a['enabled']
        return final_results

    @staticmethod
    def _construct_result_from_frames(frame1,
                                      frame2,
                                      res_type,
                                      res,
                                      label,
                                      frameCount=0,
                                      exclude_first=True):
        """
        Construct frames between 2 keyframes
        :param frame1: First frame in sequence
        :param frame2: Next frame in sequence
        :param res_type: Result type (e.g. rectanglelabels)
        :param res: Result dict
        :param label: Result label
        :param frameCount: Total frame count in the video
        :param exclude_first: Exclude first result to deduplicate results
        :return: List of frames
        """
        final_results = []
        if not frame1["enabled"]:
            return []
        if len(frame2) > 0:
            if frame1['frame'] > frame2['frame']:
                return []
            frame_count = frame2['frame'] - frame1['frame'] + 1
        else:
            frame_count = frameCount - frame1['frame'] + 1
        start_i = 1 if exclude_first else 0
        for i in range(start_i, frame_count):
            frame_number = i + frame1['frame']
            delta = i / (frame_count - 1)
            x_delta = 0 if (frame1["x"] == frame2.get("x") or frame2 == {}) else (frame2.get("x", 0) - frame1["x"]) * delta
            y_delta = 0 if (frame1["y"] == frame2.get("y") or frame2 == {}) else (frame2.get("y", 0) - frame1["y"]) * delta
            rot_delta = 0 if (frame1["rotation"] == frame2.get("rotation") or frame2 == {}) else (frame2.get("rotation", 0) - frame1["rotation"]) * delta
            width_delta = 0 if (frame1["width"] == frame2.get("width") or frame2 == {}) else (frame2.get("width", 0) - frame1["width"]) * delta
            height_delta = 0 if (frame1["height"] == frame2.get("height") or frame2 == {}) else (frame2.get("height", 0) - frame1["height"]) * delta
            result = deepcopy(res)
            result["type"] = res_type
            result["value"] = {
                    res_type: label if isinstance(label, list) else [label],
                    "x": frame1["x"] + x_delta,
                    "y": frame1["y"] + y_delta,
                    "width": frame1["width"] + width_delta,
                    "height": frame1["height"] + height_delta,
                    "rotation": frame1["rotation"] + rot_delta,
                    "frame": frame_number
                }
            if frame_number not in [frame1.get('frame'), frame2.get('frame')]:
                result["value"]["auto"] = True
            final_results.append(result)
        return final_results