from enum import Enum


class HospitalType(str, Enum):
    womens = "womens"
    children = "children"
    veterans = "veterans"
    medicare = "medicare"


class ServiceCategory(str, Enum):
    psychiatric = "psychiatric"
    medicare = "medicare"
    general = "general"
