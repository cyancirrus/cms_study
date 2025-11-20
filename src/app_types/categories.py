from enum import Enum


class HospitalType(str, Enum):
    acute = "Acute Care Hospitals"
    psych = "Psychiatric"
    critical = "Critical Access Hospitals"


class ServiceCategory(str, Enum):
    psychiatric = "psychiatry"
    medicare = "medicare"
    general = "general"
