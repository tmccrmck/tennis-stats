from prediction import Predictor

def run_sample_prediction():
    predictor = Predictor()
    
    # Predict: Sinner vs Djokovic on Hard Court (Grand Slam)
    p1 = "Jannik Sinner"
    p2 = "Novak Djokovic"
    surface = "Hard"
    level = "G"
    
    prob = predictor.predict(p1, p2, surface, level)
    
    print(f"\nConsensus Prediction (Weighted Ensemble):")
    print(f"{p1} vs {p2} on {surface} ({level})")
    print(f"Win Probability for {p1}: {prob:.2%}")
    print(f"Win Probability for {p2}: {1-prob:.2%}")

    # Predict: Alcaraz vs Zverev on Clay (Grand Slam)
    p1_alt = "Carlos Alcaraz"
    p2_alt = "Alexander Zverev"
    surf_alt = "Clay"
    
    prob_alt = predictor.predict(p1_alt, p2_alt, surf_alt, level)
    
    print(f"\nConsensus Prediction (Weighted Ensemble):")
    print(f"{p1_alt} vs {p2_alt} on {surf_alt} ({level})")
    print(f"Win Probability for {p1_alt}: {prob_alt:.2%}")
    print(f"Win Probability for {p2_alt}: {1-prob_alt:.2%}")

if __name__ == "__main__":
    run_sample_prediction()
