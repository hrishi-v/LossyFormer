from typing import Dict

from .BertEarlyExit import BertEarlyExit
from .RobertaEarlyExit import RobertaEarlyExit

EARLY_EXIT_CONSTRUCTION_MAP: Dict[str, type[BertEarlyExit] | type[RobertaEarlyExit]] = {
    "bert-base": BertEarlyExit,
    "roberta": RobertaEarlyExit,
    "bert-tiny": BertEarlyExit,
}


def get_early_exit_model(model, threshold=0.3):
    if hasattr(model, "roberta") or "roberta" in str(type(model)).lower():
        return RobertaEarlyExit(model, threshold=threshold)
    return BertEarlyExit(model, threshold=threshold)
