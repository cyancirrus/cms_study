from initialize_environment import ENGINE
from transform.executors import (
    transform_msa_for_demographics,
    transform_facility_zip_code,
)

if __name__ == "__main__":
    print("Starting transform pipeline")
    transform_msa_for_demographics(ENGINE)
    transform_facility_zip_code(ENGINE)
    print("Ending transform pipeline")
