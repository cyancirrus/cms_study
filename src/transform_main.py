from initialize_environment import ENGINE
from transform.executors import transform_msa_for_demographics

if __name__ == "__main__":
    print("Starting transform pipeline")
    transform_msa_for_demographics(ENGINE)
    print("Ending transform pipeline")
