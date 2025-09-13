import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import argparse
from typing import List, Optional
from pydantic import BaseModel
from dataset.quality.handler import QualityPredictItem

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    p_args = parser.parse_args(args)
    with open(p_args.path, 'r', encoding='utf-8') as f:
        datas = f.readlines()
    total = len(datas)
    tp = 0
    wrong_labeled: List[QualityPredictItem] = []

    for data in datas:
        data = data.strip()
        item = QualityPredictItem.model_validate_json(data)
        if item.predict_label == item.gold_label:
            tp += 1
        else:
            wrong_labeled.append(item)
    
    print(f"tp/total = {tp}/{total}  ==> {tp/total}")
    return f"tp/total = {tp}/{total}  ==> {tp/total}"
    

if __name__ == "__main__":
    main()