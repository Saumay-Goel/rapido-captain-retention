# Data Assumptions & Sources

## Rapido Captain Churn — Simulation Documentation

---

## 1. Why We Did Not Use the Kaggle Dataset

The only public Rapido dataset (kaggle.com/vengateshvengat/rapido-all-data)
was verified to be low-quality synthetic data:

- Cancellation rate identical across all service types (~10%)
- Fare per km identical across all service types (~Rs.43)
- Avg duration identical across all service types (~64 mins)
- Payment methods split exactly 25% each — statistically impossible in real life

Decision: Dataset rejected. Simulation built from scratch using primary research.

---

## 2. Captain Identity Assumptions

| Feature                 | Value                                      | Source                               |
| ----------------------- | ------------------------------------------ | ------------------------------------ |
| Cities covered          | Bengaluru, Hyderabad, Chennai, Pune, Delhi | Rapido operational cities            |
| City weight (Bengaluru) | 35% of captains                            | Rapido HQ + largest market           |
| Vehicle split           | 72% bike, 28% auto                         | Rapido product mix, news reports     |
| Captain type split      | 40% fulltime, 60% parttime                 | Reddit: most treat it as side hustle |

---

## 3. Ride Volume Assumptions

| Feature                 | Value                      | Source                              |
| ----------------------- | -------------------------- | ----------------------------------- |
| Fulltime rides per week | ~80 (40-120 range)         | Reddit: 10-15 rides/day target      |
| Parttime rides per week | ~28 (5-50 range)           | Reddit: 4-6 rides/day part-timers   |
| Week over week decay    | 45-95% retention each week | Observed dropout pattern from vlogs |

Source: Reddit r/bangalore post — captain rode 4 days, ~5hrs/day, earned Rs.2,220 gross

---

## 4. Earnings Assumptions

| Feature                   | Value                | Source                               |
| ------------------------- | -------------------- | ------------------------------------ |
| Bike fare per km          | Rs.7-16 (avg Rs.11)  | Reddit captain vlogs                 |
| Auto fare per km          | Rs.13-26 (avg Rs.19) | Common knowledge + Reddit            |
| Night fare bonus          | 20% premium 10PM-6AM | Reddit r/bangalore captain post 2024 |
| Net earnings after petrol | ~Rs.3/km             | Reddit r/noida captain report        |

---

## 5. Behavioral Feature Assumptions

| Feature                | Value     | Source                                      |
| ---------------------- | --------- | ------------------------------------------- |
| Bike cancellation rate | 8-22%     | Industry reports                            |
| Auto cancellation rate | 4-14%     | Industry reports                            |
| Streak completion rate | 27%       | Reddit: "8 rides in 4hrs is hard"           |
| Incentive claim rate   | 52%       | Estimated from platform behavior            |
| Avg bike ride duration | 17 mins   | Reddit captain vlogs                        |
| Avg auto ride duration | 24 mins   | Reddit captain vlogs                        |
| Zone switches fulltime | ~1.5/week | Reddit: experienced captains stick to zones |
| Zone switches parttime | ~3.0/week | Reddit: new captains switch desperately     |

---

## 6. Payment Method Assumptions

| Method     | Share | Source                            |
| ---------- | ----- | --------------------------------- |
| GPay       | 40%   | NPCI UPI market share report 2024 |
| Paytm      | 20%   | Dropped after RBI action 2024     |
| Amazon Pay | 18%   | Estimated                         |
| QR Scan    | 12%   | Estimated                         |
| Cash       | 10%   | Estimated                         |

---

## 7. Churn Label Assumptions

Churn is defined as: inactive for 30+ days after Week 4

Churn probability driven by (in order of importance per SHAP):

1. Rides in Week 4 — strongest signal
2. Estimated daily earnings — low earnings = dropout
3. Cancellation rate — high cancellation = low earnings = churn
4. Incentive claimed — protective factor
5. Streak completed — protective factor
6. Zone switches — desperation signal
7. Captain type — parttime churn baseline higher
8. Petrol cost sensitivity — thin margin pressure

Overall churn rate: 41% — consistent with Rapido's publicly stated
early captain retention challenges

---

## 8. What This Simulation Does Not Capture

Being transparent about limitations:

- No seasonal effects (festival demand spikes)
- No geographic micro-zone data (lane level demand)
- No actual platform incentive structure details (proprietary)
- No competitor switching data (Ola, Uber, Namma Yatri)
- Churn label is simulated, not observed from real platform logs

---

## 9. Primary Sources Used

- Reddit r/bangalore — captain experience posts (2024)
- Reddit r/noida, r/chennai, r/pune — captain earning reports
- Rapido captain YouTube vlogs (search: "Rapido captain earning")
- NPCI UPI market share dashboard (2024)
- MoRTH Road Transport Yearbook (2023)
- Rapido press releases on captain onboarding
- IDInsight gig worker India survey reports
