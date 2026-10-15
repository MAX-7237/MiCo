"""Official RefCOCO/+/g referring-expression comprehension evaluation."""

import json
import math
import os
import pickle
import re
from collections import defaultdict

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import LMUDataRoot, dump, load, osp


class RefCOCOGroundingDataset(ImageBaseDataset):
    """Evaluate one generated box per referring expression with Acc@IoU=0.5."""

    MODALITY = 'IMAGE'
    # Keep this outside VQA so Qwen's generic short-answer prompt cannot replace
    # the coordinate-only prompt required by the referring-expression protocol.
    TYPE = 'GROUNDING'
    IOU_THRESHOLD = 0.5
    DATASETS = {
        'RefCOCO': ('refcoco', 'refs(unc).p', ('val', 'testA', 'testB')),
        'RefCOCO+': ('refcoco+', 'refs(unc).p', ('val', 'testA', 'testB')),
        # UMD is the standard RefCOCOg variant with both validation and test splits.
        'RefCOCOg': ('refcocog', 'refs(umd).p', ('val', 'test')),
    }
    _JSON_BBOX = re.compile(
        r'"bbox_2d"\s*:\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*'
        r'([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*'
        r'([-+]?\d+(?:\.\d+)?)\s*\]'
    )
    _TUPLE_BBOX = re.compile(
        r'\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)'
        r'\s*,?\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*'
        r'([-+]?\d+(?:\.\d+)?)\s*\)'
    )

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASETS)

    def __init__(self, dataset='RefCOCO', skip_noimg=True):
        if dataset not in self.DATASETS:
            raise ValueError(f'Unsupported RefCOCO dataset: {dataset}')
        self.dataset_name = dataset
        variant, refs_file, splits = self.DATASETS[dataset]
        self.ref_root = osp.join(LMUDataRoot(), 'RefCOCO_suite')
        self.img_root = osp.join(self.ref_root, 'images', 'mscoco', 'images', 'train2014')

        with open(osp.join(self.ref_root, variant, 'instances.json'), encoding='utf-8') as handle:
            instances = json.load(handle)
        with open(osp.join(self.ref_root, variant, refs_file), 'rb') as handle:
            refs = pickle.load(handle, encoding='latin1')

        images = {item['id']: item for item in instances['images']}
        annotations = {item['id']: item for item in instances['annotations']}
        rows = []
        required_images = set()
        for ref in refs:
            if ref['split'] not in splits:
                continue
            annotation = annotations.get(ref['ann_id'])
            image = images.get(ref['image_id'])
            if annotation is None or image is None:
                raise ValueError(f'{dataset}: missing COCO record for ref_id={ref["ref_id"]}')
            image_path = osp.join(self.img_root, image['file_name'])
            required_images.add(image_path)
            x, y, width, height = annotation['bbox']
            for sentence in ref['sentences']:
                rows.append({
                    'index': f'{dataset}:{ref["split"]}:{ref["ref_id"]}:{sentence["sent_id"]}',
                    'split': ref['split'],
                    'question': sentence['sent'],
                    'image_path': image_path,
                    'image_width': int(image['width']),
                    'image_height': int(image['height']),
                    'bbox_xyxy': [float(x), float(y), float(x + width), float(y + height)],
                })

        missing = [path for path in required_images if not osp.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f'{dataset}: {len(missing)} required COCO train2014 images are absent under {self.img_root}; '
                f'first missing: {missing[0]}'
            )
        self.data = pd.DataFrame(rows)
        if self.data.empty or set(self.data['split']) != set(splits):
            raise ValueError(f'{dataset}: incomplete official split set {splits}')
        smoke_rows = int(os.environ.get('TOPS_EXPERIMENTAL_SMOKE_ROWS', '0'))
        if smoke_rows > 0:
            # A smoke run validates the model-specific parser before spending GPUs on every expression.
            self.data = self.data.iloc[:smoke_rows].copy()
        self.meta_only = True

    def dump_image(self, line):
        return [str(line['image_path'])]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mode = os.environ.get('REFCOCO_COORD_MODE')
        expression = str(line['question'])
        if mode == 'qwen25_resized_absolute':
            resized_height, resized_width = self._qwen25_resized_shape(
                int(line['image_height']), int(line['image_width'])
            )
            coordinate_instruction = (
                f'Report coordinates in pixels for the {resized_width} by {resized_height} '
                'resized image used by the model, not the original image.'
            )
        elif mode == 'qwen3_relative_1000':
            coordinate_instruction = (
                'Report coordinates as integers from 0 to 1000, where [0, 0] is the top-left '
                'and [1000, 1000] is the bottom-right of the image.'
            )
        else:
            raise RuntimeError('REFCOCO_COORD_MODE must be qwen25_resized_absolute or qwen3_relative_1000')
        prompt = (
            f'Locate the region described by: "{expression}". {coordinate_instruction} '
            'Return exactly one JSON object and nothing else: '
            '{"bbox_2d": [x1, y1, x2, y2]}.'
        )
        return [
            {'type': 'image', 'value': str(line['image_path'])},
            {'type': 'text', 'value': prompt},
        ]

    @classmethod
    def _parse_bbox(cls, response):
        match = cls._JSON_BBOX.search(str(response))
        if match is None:
            match = cls._TUPLE_BBOX.search(str(response))
        if match is None:
            return None
        bbox = [float(value) for value in match.groups()]
        return bbox if all(math.isfinite(value) for value in bbox) else None

    @staticmethod
    def _iou(predicted, target):
        left = max(predicted[0], target[0])
        top = max(predicted[1], target[1])
        right = min(predicted[2], target[2])
        bottom = min(predicted[3], target[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        predicted_area = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
        target_area = (target[2] - target[0]) * (target[3] - target[1])
        union = predicted_area + target_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _valid_box(box, width, height):
        return (
            box is not None
            and 0 <= box[0] < box[2] <= width
            and 0 <= box[1] < box[3] <= height
        )

    @staticmethod
    def _qwen25_resized_shape(height, width):
        from qwen_vl_utils import smart_resize

        # Mirror the Qwen2.5 wrapper's preprocessing so model boxes and IoU share one frame.
        return smart_resize(
            height, width, factor=28,
            min_pixels=1280 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        )

    @classmethod
    def _to_original_scale(cls, box, width, height, mode):
        if mode == 'qwen3_relative_1000':
            return [box[0] * width / 1000.0, box[1] * height / 1000.0,
                    box[2] * width / 1000.0, box[3] * height / 1000.0]
        if mode == 'qwen25_resized_absolute':
            resized_height, resized_width = cls._qwen25_resized_shape(height, width)
            return [box[0] * width / resized_width, box[1] * height / resized_height,
                    box[2] * width / resized_width, box[3] * height / resized_height]
        raise RuntimeError(f'Unsupported coordinate mode: {mode}')

    def evaluate(self, eval_file, **judge_kwargs):
        mode = os.environ.get('REFCOCO_COORD_MODE')
        if mode not in {'qwen25_resized_absolute', 'qwen3_relative_1000'}:
            raise RuntimeError('REFCOCO_COORD_MODE is required for RefCOCO scoring')
        predictions = load(eval_file)
        if 'index' not in predictions or 'prediction' not in predictions:
            raise ValueError('RefCOCO predictions must contain index and prediction columns')
        predictions['index'] = predictions['index'].astype(str)
        if predictions['index'].duplicated().any():
            raise ValueError('RefCOCO prediction indices must be unique')
        prediction_map = dict(zip(predictions['index'], predictions['prediction']))
        expected_indices = set(self.data['index'].astype(str))
        missing_indices = expected_indices.difference(prediction_map)
        # A partial distributed output is not a valid benchmark score, even if run.py exits zero.
        if missing_indices:
            raise ValueError(f'RefCOCO predictions are incomplete: missing {len(missing_indices)} expressions')

        grouped = defaultdict(list)
        details = []
        for _, row in self.data.iterrows():
            raw_response = prediction_map.get(str(row['index']), '')
            raw_box = self._parse_bbox(raw_response)
            predicted = None
            if raw_box is not None:
                predicted = self._to_original_scale(
                    raw_box, int(row['image_width']), int(row['image_height']), mode
                )
            valid = self._valid_box(predicted, float(row['image_width']), float(row['image_height']))
            iou = self._iou(predicted, row['bbox_xyxy']) if valid else 0.0
            if not math.isfinite(iou):
                raise ValueError(f'Non-finite IoU for {row["index"]}')
            record = {
                'index': row['index'],
                'split': row['split'],
                'raw_response': str(raw_response),
                'raw_bbox': raw_box,
                'pred_bbox_xyxy': predicted,
                'gt_bbox_xyxy': row['bbox_xyxy'],
                'iou': iou,
                'hit_at_0_5': bool(iou >= self.IOU_THRESHOLD),
            }
            grouped[row['split']].append(record)
            details.append(record)

        def summarize(records):
            count = len(records)
            if count == 0:
                raise ValueError('A RefCOCO split has no evaluated expressions')
            return {
                'n': count,
                'parse_rate': sum(item['raw_bbox'] is not None for item in records) / count,
                'mean_iou': sum(item['iou'] for item in records) / count,
                'acc_at_iou_0_5': sum(item['hit_at_0_5'] for item in records) / count,
            }

        summary = {
            'dataset': self.dataset_name,
            'coordinate_mode': mode,
            'metric': 'Acc@IoU>=0.5',
            'splits': {split: summarize(records) for split, records in sorted(grouped.items())},
            'overall': summarize(details),
        }
        stem, _ = osp.splitext(eval_file)
        score_file = f'{stem}_score.json'
        detail_file = f'{stem}_grounding.jsonl'
        dump(summary, score_file)
        pd.DataFrame(details).to_json(detail_file, orient='records', lines=True, force_ascii=False)
        return summary
