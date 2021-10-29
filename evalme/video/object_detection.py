from evalme.eval_item import EvalItem

class VideoObjectDetectionEvalItem(EvalItem):
    SHAPE_KEY = 'videorectanglelabels'

    def extract_key_frames(self):
        final_results = []
        for result in self._raw_data:
            sequence = result['value']['sequence']
            if len(sequence) < 1:
                continue
            label = result['value']['labels']
            sequence = sorted(sequence, key=lambda d: d['frame'])
            if len(sequence) < 2:
                element = sequence.pop()
                final_results.append(
                    {
                        "id": result['id'],
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": label if isinstance(label, list) else [label],
                            "x": element["x"],
                            "y": element["y"],
                            "width": element["width"],
                            "height": element["height"],
                            "rotation": element["rotation"],
                            "frame": element["frame"]
                        }
                    }
                )
            else:
                for i in range(len(sequence)-1):
                    frame_a = sequence[i]
                    frame_b = sequence[i+1]
                    final_results.extend(self._construct_result_from_frames(frame_a,
                                                                            frame_b,
                                                                            "rectanglelabels",
                                                                            result['id'],
                                                                            label))
        return final_results

    @staticmethod
    def _construct_result_from_frames(frame1, frame2, res_type, res_id, label):
        final_results = []
        if frame1['frame'] > frame2['frame'] or (not frame1["enabled"]):
            return []
        frame_count = frame2['frame'] - frame1['frame'] + (1 if frame2["enabled"] else 0) # including both ends if frame2 enabled
        for i in range(frame_count):
            new_frame = {}
            frame_number = i + frame1['frame']
            not_moving = frame1["x"] == frame2["x"] and frame1["y"] == frame2["y"] and frame1["rotation"] == frame2["rotation"]
            not_changing = frame1["width"] == frame2["width"] and frame1["height"] == frame2["height"]
            new_frame["x"] = frame1["x"] if not_moving \
                else (frame1["x"] + (frame2["x"] - frame1["x"]) * i / (frame_count - 1))
            new_frame["y"] = frame1["y"] if not_moving \
                else (frame1["y"] + (frame2["y"] - frame1["y"]) * i / (frame_count - 1))
            new_frame["rotation"] = frame1["rotation"] if not_moving \
                else (frame1["rotation"] + (frame2["rotation"] - frame1["rotation"]) * i / (frame_count - 1))
            new_frame["width"] = frame1["width"] if not_changing \
                else (frame1["width"] + (frame2["width"] - frame1["width"]) * i / (frame_count - 1))
            new_frame["height"] = frame1["height"] if not_changing \
                else (frame1["height"] + (frame2["height"] - frame1["height"]) * i / (frame_count - 1))
            result = {
                "id": res_id,
                "type": res_type,
                "value": {
                    res_type: label if isinstance(label, list) else [label],
                    "x": new_frame["x"],
                    "y": new_frame["y"],
                    "width": new_frame["width"],
                    "height": new_frame["height"],
                    "rotation": new_frame["rotation"],
                    "frame": frame_number
                }
            }
            final_results.append(result)
        return final_results
