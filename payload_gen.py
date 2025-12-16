"""Generate a sample payload.json file for testing the prediction API."""
import json

feature_names = [
    # paste the 55 feature names here as Python strings, e.g.:
    # "Amount_sum",
    # "Amount_mean",
    # ...
    "Amount_sum",
    "Amount_mean",
    "Amount_max",
    "Amount_std",
    "Amount_skew",
    "Value_sum",
    "Value_mean",
    "Value_max",
    "Value_std",
    "Value_skew",
    "TransactionStartTime_count",
    "TransactionStartTime_ < lambda_0 >",
    "Hour_mean",
    "Hour_std",
    "Weekday_mean",
    "Weekday_std",
    "Day_mean",
    "Day__first_mode_or_nan",
    "Month_mean",
    "Month__first_mode_or_nan",
    "Year_mean",
    "Year__first_mode_or_nan",
    "IsWeekend_mean",
    "CurrencyCode_UGX",
    "ProviderId_ProviderId_1",
    "ProviderId_ProviderId_2",
    "ProviderId_ProviderId_3",
    "ProviderId_ProviderId_4",
    "ProviderId_ProviderId_5",
    "ProviderId_ProviderId_6",
    "ChannelId_ChannelId_1",
    "ChannelId_ChannelId_2",
    "ChannelId_ChannelId_3",
    "ChannelId_ChannelId_5",
    "TransactionId_freq",
    "BatchId_freq",
    "AccountId_freq",
    "SubscriptionId_freq",
    "ProductId_freq",
    "ProductCategory_freq",
    "Amount_sum_log_std",
    "Amount_mean_log_std",
    "Amount_max_log_std",
    "Amount_skew_log_std",
    "Value_mean_log_std",
    "Value_skew_log_std",
    "TransactionId_woe",
    "BatchId_woe",
    "AccountId_woe",
    "SubscriptionId_woe",
    "CurrencyCode_woe",
    "ProviderId_woe",
    "ProductId_woe",
    "ProductCategory_woe",
    "ChannelId_woe"
]

payload = {
    "customer_id": "smoke_test_1",
    "features": {name: 0.0 for name in feature_names}
}

with open("payload.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
print("Wrote payload.json with", len(feature_names), "features")
