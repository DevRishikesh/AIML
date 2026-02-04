import random

# Observed data (Traffic only)
data = ["Heavy", "Light", "Heavy", "Heavy", "Light"]

# Initialize parameters
P_rain = 0.5  # P(Rain = Yes)

P_traffic_given_rain = {
    "Yes": {"Heavy": 0.6, "Light": 0.4},
    "No": {"Heavy": 0.3, "Light": 0.7}
}

# EM Algorithm
iterations = 10

for it in range(iterations):

    # E-step: Expected counts
    expected_rain_yes = 0
    expected_rain_no = 0

    traffic_counts = {
        "Yes": {"Heavy": 0, "Light": 0},
        "No": {"Heavy": 0, "Light": 0}
    }

    for t in data:
        # Bayes rule
        prob_yes = P_rain * P_traffic_given_rain["Yes"][t]
        prob_no = (1 - P_rain) * P_traffic_given_rain["No"][t]

        total = prob_yes + prob_no
        prob_yes /= total
        prob_no /= total

        expected_rain_yes += prob_yes
        expected_rain_no += prob_no

        traffic_counts["Yes"][t] += prob_yes
        traffic_counts["No"][t] += prob_no

    # M-step: Update parameters
    P_rain = expected_rain_yes / len(data)

    for rain in ["Yes", "No"]:
        total = sum(traffic_counts[rain].values())
        for t in ["Heavy", "Light"]:
            P_traffic_given_rain[rain][t] = traffic_counts[rain][t] / total

# Output
print("Final Estimated Parameters:")
print("P(Rain = Yes):", round(P_rain, 3))
print("P(Traffic | Rain):")
print(P_traffic_given_rain)
