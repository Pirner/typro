from dataclasses import dataclass


@dataclass
class DataPoint:
    im_path: str
    label: int
