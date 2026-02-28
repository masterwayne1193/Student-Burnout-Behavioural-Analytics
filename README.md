# Student-Burnout-Behavioural-Analytics

# Early Detection of Student Burnout & Dropout Risk

## Dataset Type
Synthetic dataset (2000 records)

## Why Synthetic?
No real behavioural university dataset available due to privacy constraints.

## Data Generation Logic
Features generated using normal and uniform distributions.
Risk score computed using behavioural weighting logic:
- Lower LMS activity increases risk
- Higher assignment delays increase risk
- Lower attendance increases risk
- Negative sentiment increases risk
- High irregular activity increases risk
- Late-night usage increases stress indicator

## Features
- lms_login_freq
- assignment_delay_days
- attendance_percent
- sentiment_score
- activity_irregularity
- late_night_usage
- risk_score
- burnout_level
- dropout

## Models Used
- Random Forest (Burnout classification)
- Logistic Regression (Dropout probability)
- Decision Tree (Behaviour explanation)

## Output
- Risk Score (0–100 scaled)
- Burnout Level (Low/Medium/High)
- Dropout Probability
- Behavioural Triggers
- Intervention Suggestions
