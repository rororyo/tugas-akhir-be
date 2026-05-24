import requests
import json

API_URL = "http://localhost:5000"

# ---------------------------------------------------------------------------
# Test transactions using the current model schema.
#
# Required fields : amount, merchant_category
# Optional fields : user_id, transaction_time, channel, account_age_days,
#                   shipping_distance_km, avs_match, cvv_result,
#                   three_ds_flag, promo_used, country, bin_country
#
# Valid merchant_category values : electronics, travel, grocery, gaming, fashion
# Valid channel values            : web, app
# Valid country / bin_country     : US, GB, FR, NL, TR, PL, RO, DE, ES, IT
# avs_match / cvv_result / three_ds_flag / promo_used : 1 = yes, 0 = no
# ---------------------------------------------------------------------------

TEST_TRANSACTIONS = [
    # 1 – Low-risk grocery order, established account, all security checks pass
    {
        "user_id": 1001,
        "amount": 45.99,
        "merchant_category": "grocery",
        "channel": "app",
        "account_age_days": 730,
        "country": "US",
        "bin_country": "US",
        "avs_match": 1,
        "cvv_result": 1,
        "three_ds_flag": 1,
        "promo_used": 0,
        "shipping_distance_km": 12.5,
        "transaction_time": "2024-03-15T14:30:00Z",
        "expected": 0,
    },
    # 2 – Fashion order with promo, medium-age account, matches expected profile
    {
        "user_id": 1002,
        "amount": 189.50,
        "merchant_category": "fashion",
        "channel": "web",
        "account_age_days": 365,
        "country": "US",
        "bin_country": "US",
        "avs_match": 1,
        "cvv_result": 1,
        "three_ds_flag": 1,
        "promo_used": 1,
        "shipping_distance_km": 25.0,
        "transaction_time": "2024-02-20T10:15:00Z",
        "expected": 0,
    },
    # 3 – Travel purchase, cross-country card mismatch, no OTP, late night
    {
        "user_id": 1003,
        "amount": 850.00,
        "merchant_category": "travel",
        "channel": "web",
        "account_age_days": 45,
        "country": "FR",
        "bin_country": "TR",
        "avs_match": 0,
        "cvv_result": 1,
        "three_ds_flag": 0,
        "promo_used": 0,
        "shipping_distance_km": 1200.0,
        "transaction_time": "2024-01-10T02:30:00Z",
        "expected": 1,
    },
    # 4 – High-value electronics, brand-new account, all security checks fail
    {
        "user_id": 1004,
        "amount": 1299.99,
        "merchant_category": "electronics",
        "channel": "web",
        "account_age_days": 15,
        "country": "US",
        "bin_country": "RO",
        "avs_match": 0,
        "cvv_result": 0,
        "three_ds_flag": 0,
        "promo_used": 1,
        "shipping_distance_km": 450.0,
        "transaction_time": "2024-04-05T03:45:00Z",
        "expected": 1,
    },
    # 5 – Gaming digital goods, mid-age account, good security signals
    {
        "user_id": 1005,
        "amount": 89.99,
        "merchant_category": "gaming",
        "channel": "app",
        "account_age_days": 180,
        "country": "GB",
        "bin_country": "GB",
        "avs_match": 1,
        "cvv_result": 1,
        "three_ds_flag": 0,
        "promo_used": 0,
        "shipping_distance_km": 0.0,
        "transaction_time": "2024-03-22T16:00:00Z",
        "expected": 0,
    },
    # 6 – Micro-amount card-testing pattern, very new account, no security
    {
        "user_id": 1006,
        "amount": 1.99,
        "merchant_category": "electronics",
        "channel": "web",
        "account_age_days": 5,
        "country": "US",
        "bin_country": "PL",
        "avs_match": 0,
        "cvv_result": 0,
        "three_ds_flag": 0,
        "promo_used": 0,
        "shipping_distance_km": 800.0,
        "transaction_time": "2024-01-25T01:00:00Z",
        "expected": 1,
    },
    # 7 – Routine grocery, long-standing account, promo + OTP, short distance
    {
        "user_id": 1007,
        "amount": 67.40,
        "merchant_category": "grocery",
        "channel": "app",
        "account_age_days": 600,
        "country": "US",
        "bin_country": "US",
        "avs_match": 1,
        "cvv_result": 1,
        "three_ds_flag": 1,
        "promo_used": 1,
        "shipping_distance_km": 5.2,
        "transaction_time": "2024-02-14T12:00:00Z",
        "expected": 0,
    },
    # 8 – Large travel booking, very new account, extreme distance, no OTP
    {
        "user_id": 1008,
        "amount": 2500.00,
        "merchant_category": "travel",
        "channel": "web",
        "account_age_days": 8,
        "country": "DE",
        "bin_country": "TR",
        "avs_match": 0,
        "cvv_result": 1,
        "three_ds_flag": 0,
        "promo_used": 0,
        "shipping_distance_km": 2500.0,
        "transaction_time": "2024-03-01T23:50:00Z",
        "expected": 1,
    },
]


def test_batch_json():
    """POST /predict — JSON batch (used by the frontend file-upload flow)."""
    print("\n" + "=" * 85)
    print("TEST: POST /predict  (JSON batch)")
    print(f"{'#':<4} {'USER':<8} {'CATEGORY':<14} {'AMOUNT':>10} {'PROB':>8} {'PREDICTED':<12} {'EXPECTED':<10} RESULT")
    print("-" * 85)

    transactions = [{k: v for k, v in tx.items() if k != "expected"} for tx in TEST_TRANSACTIONS]
    expected_labels = [tx["expected"] for tx in TEST_TRANSACTIONS]

    payload = {"transactions": transactions}

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"❌  Cannot connect to {API_URL} — is the server running?")
        return

    if resp.status_code != 200:
        print(f"❌  API error {resp.status_code}: {resp.text}")
        return

    predictions = resp.json()["predictions"]
    correct = 0

    for i, pred in enumerate(predictions):
        expected = expected_labels[i]
        tx       = TEST_TRANSACTIONS[i]
        predicted = 1 if pred["is_fraud"] else 0
        prob      = pred["fraud_probability"]
        match     = "✅ MATCH" if predicted == expected else "❌ MISMATCH"
        if predicted == expected:
            correct += 1
        print(
            f"{i+1:<4} {tx['user_id']:<8} {tx['merchant_category']:<14} "
            f"{tx['amount']:>10.2f} {prob:>8.4f} "
            f"{'Fraud' if pred['is_fraud'] else 'Safe':<12} "
            f"{'Fraud' if expected else 'Safe':<10} {match}"
        )

    total = len(predictions)
    print("-" * 85)
    print(f"Accuracy: {correct}/{total}  ({correct/total*100:.1f}%)")
    print(f"Threshold used: {resp.json()['threshold']:.4f}")


def test_single():
    """POST /api/predict/single — single JSON transaction (form mode)."""
    print("\n" + "=" * 85)
    print("TEST: POST /api/predict/single  (form / single mode)")
    print("-" * 85)

    body = {
        "user_id": 9001,
        "amount": 349.00,
        "merchant_category": "electronics",
        "channel": "web",
        "account_age_days": 22,
        "country": "US",
        "bin_country": "NL",
        "avs_match": 0,
        "cvv_result": 0,
        "three_ds_flag": 0,
        "promo_used": 1,
        "shipping_distance_km": 600.0,
        "transaction_time": "2024-05-10T03:15:00Z",
    }

    try:
        resp = requests.post(f"{API_URL}/api/predict/single", json=body, timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"❌  Cannot connect to {API_URL} — is the server running?")
        return

    if resp.status_code != 200:
        print(f"❌  API error {resp.status_code}: {resp.text}")
        return

    result = resp.json()
    print(json.dumps(result, indent=2))


def test_health():
    """GET /api/health."""
    print("\n" + "=" * 85)
    print("TEST: GET /api/health")
    print("-" * 85)
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=10)
        print(json.dumps(resp.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print(f"❌  Cannot connect to {API_URL}")


if __name__ == "__main__":
    test_health()
    test_batch_json()
    test_single()
