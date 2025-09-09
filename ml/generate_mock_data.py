# ml/generate_mock_data.py
import pandas as pd
import random
crops = ["Wheat","Rice","Maize","Cotton","Soybean","Pulses"]
rows = []
for i in range(500):
    pH = round(random.uniform(5.2,8.0),2)
    N = random.randint(100,400)
    P = random.randint(10,80)
    K = random.randint(50,300)
    rainfall = random.randint(100,800) # mm/season
    temp = random.uniform(18,33)
    prev = random.choice(crops)
    market_price = random.randint(900,4000) # ₹/quintal
    # simplistic logic for target (for mock only)
    if rainfall>500 and pH>6.0:
        rec = "Rice"
    elif pH<6.0 and N>200:
        rec = "Wheat"
    else:
        rec = random.choice(crops)
    rows.append([pH,N,P,K,rainfall,temp,prev,market_price,rec])
df = pd.DataFrame(rows, columns=["soil_pH","nitrogen","phosphorus","potassium","rainfall","temperature","previous_crop","market_price","recommended_crop"])
df.to_csv("data/mock_soil_market.csv", index=False)
print("Saved data/mock_soil_market.csv")
