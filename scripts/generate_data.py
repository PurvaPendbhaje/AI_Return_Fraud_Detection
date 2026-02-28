import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

rows = 1200

data = pd.DataFrame({
    "returns_last_90_days": np.random.randint(0,15,rows),
    "avg_refund_amount": np.random.randint(200,8000,rows),
    "purchase_return_gap_days": np.random.randint(1,30,rows),
    "damage_claim_frequency": np.random.randint(0,10,rows),
    "high_value_item": np.random.randint(0,2,rows),
    "return_reason_similarity": np.random.rand(rows),
    "return_purchase_ratio": np.random.rand(rows)
})

data["fraud_label"] = np.where(
    (data["returns_last_90_days"]>7) &
    (data["damage_claim_frequency"]>4) &
    (data["avg_refund_amount"]>3000),
    1,0
)

data.to_csv("data/returns.csv",index=False)

print("Dataset created successfully!")