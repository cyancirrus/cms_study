from initialize_environment import ENGINE, CURRENT_YEAR
from recommendation.hospital import build_recommendation_table

if __name__ == "__main__":
    print("---------------------------------------")
    print("   Building Recommendation Table       ")
    print("---------------------------------------")
    build_recommendation_table(ENGINE, CURRENT_YEAR)
