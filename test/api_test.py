import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

def test_batch_complete():
    """Testing with complete transaction data including all features"""
    print("=" * 90)
    print(f"{'TX ID':<15} | {'EXPECTED':<10} | {'PREDICTED':<10} | {'PROBABILITY':<12} | {'RESULT'}")
    print("-" * 90)
    
    # Complete test data with ALL features from your CSV
    test_data = [
        {
            "Transaction ID": "a858dc67-2d9a-48d9-b89a-1a14cefa26c6",
            "Transaction Amount": 110,
            "Transaction Date": "2025-03-21 12:00:00",
            "Payment Method": "PayPal",
            "Product Category": "clothing",
            "Quantity": 3,
            "Customer Age": 30,
            "Customer Location": "Stephenfort",
            "Device Used": "mobile",
            "IP Address": "172.110.189.188",
            "Shipping Address": "0899 Jonathan Islands Kellermouth, CO 07405",
            "Billing Address": "0899 Jonathan Islands Kellermouth, CO 07405",
            "Account Age Days": 182,
            "Transaction Hour": 12,
            "expected": 1
        },
        {
            "Transaction ID": "fcd06010-535d-48db-a95a-a36b44782144",
            "Customer ID": "84652eb9-b6d7-44a3-8ba3-6653a165f645",
            "Transaction Amount": 116.98,
            "Transaction Date": "2024-03-30 11:48:55",
            "Payment Method": "PayPal",
            "Product Category": "clothing",
            "Quantity": 3,
            "Customer Age": 49,
            "Customer Location": "Parkshaven",
            "Device Used": "mobile",
            "IP Address": "213.101.171.225",
            "Shipping Address": "6139 Heather Branch Apt. 075 West Alyssa, CA 26529",
            "Billing Address": "6139 Heather Branch Apt. 075 West Alyssa, CA 26529",
            "Account Age Days": 174,
            "Transaction Hour": 11,
            "expected": 0
        },
        {
            "Transaction ID": "710aef60-ff0b-4767-8f0a-fa4a72044a3a",
            "Customer ID": "519d9b03-5049-47fa-bce9-15d159bf856f",
            "Transaction Amount": 1465.65,
            "Transaction Date": "2024-02-21 15:04:45",
            "Payment Method": "debit card",
            "Product Category": "toys & games",
            "Quantity": 5,
            "Customer Age": 58,
            "Customer Location": "East John",
            "Device Used": "desktop",
            "IP Address": "7.31.74.212",
            "Shipping Address": "8805 David Union Suite 461 Staceyshire, MP 32513",
            "Billing Address": "8805 David Union Suite 461 Staceyshire, MP 32513",
            "Account Age Days": 141,
            "Transaction Hour": 2,
            "expected": 1
        },
        {
            "Transaction ID": "a44ee3c4-e2ec-45f8-be5a-5c1775257b06",
            "Customer ID": "8fa7c2fb-5b9e-4216-a59d-88c1d270ebe3",
            "Transaction Amount": 95.05,
            "Transaction Date": "2024-01-26 06:32:30",
            "Payment Method": "PayPal",
            "Product Category": "home & garden",
            "Quantity": 1,
            "Customer Age": 45,
            "Customer Location": "Jordanburgh",
            "Device Used": "tablet",
            "IP Address": "73.211.181.175",
            "Shipping Address": "9250 Hayes Row Apt. 209 New Johnhaven, NC 36788",
            "Billing Address": "9250 Hayes Row Apt. 209 New Johnhaven, NC 36788",
            "Account Age Days": 184,
            "Transaction Hour": 6,
            "expected": 0
        },
        {
            "Transaction ID": "b762e358-7818-4f8d-86fb-353957d9c74a",
            "Customer ID": "6f76d257-e40f-49be-978a-6c5bab613e1e",
            "Transaction Amount": 16.28,
            "Transaction Date": "2024-03-15 16:23:28",
            "Payment Method": "PayPal",
            "Product Category": "toys & games",
            "Quantity": 4,
            "Customer Age": 30,
            "Customer Location": "Nathanmouth",
            "Device Used": "desktop",
            "IP Address": "195.89.97.75",
            "Shipping Address": "72231 Simmons Bridge Velasquezstad, CO 53843",
            "Billing Address": "72231 Simmons Bridge Velasquezstad, CO 53843",
            "Account Age Days": 94,
            "Transaction Hour": 16,
            "expected": 0
        },
        {
            "Transaction ID": "9651adef-8b42-48be-88d4-88a4c2933b54",
            "Customer ID": "532a3f51-afa7-4e29-a927-fb5d7f1f5eed",
            "Transaction Amount": 58.93,
            "Transaction Date": "2024-03-08 17:49:54",
            "Payment Method": "PayPal",
            "Product Category": "home & garden",
            "Quantity": 5,
            "Customer Age": 31,
            "Customer Location": "New Erik",
            "Device Used": "tablet",
            "IP Address": "37.73.95.140",
            "Shipping Address": "1427 Christopher Ridge Suite 465 Michaelmouth, FM 09795",
            "Billing Address": "1427 Christopher Ridge Suite 465 Michaelmouth, FM 09795",
            "Account Age Days": 51,
            "Transaction Hour": 17,
            "expected": 0
        },
        {
            "Transaction ID": "8785f4cc-8c08-4878-8ba4-ac50cd8eaeac",
            "Customer ID": "05e1d887-632e-427a-8320-e7d4bc07b422",
            "Transaction Amount": 101.32,
            "Transaction Date": "2024-03-23 05:40:37",
            "Payment Method": "debit card",
            "Product Category": "health & beauty",
            "Quantity": 1,
            "Customer Age": 27,
            "Customer Location": "Leslietown",
            "Device Used": "desktop",
            "IP Address": "145.105.40.251",
            "Shipping Address": "4529 Murphy Inlet Apt. 319 Stacyborough, AR 12397",
            "Billing Address": "4529 Murphy Inlet Apt. 319 Stacyborough, AR 12397",
            "Account Age Days": 47,
            "Transaction Hour": 5,
            "expected": 0
        },
        {
            "Transaction ID": "c69d66cc-6f50-4c74-893d-8f321ee2939c",
            "Customer ID": "5a5c347c-5a01-40b2-bce7-5ff510210aee",
            "Transaction Amount": 62.18,
            "Transaction Date": "2024-03-21 19:48:54",
            "Payment Method": "bank transfer",
            "Product Category": "clothing",
            "Quantity": 2,
            "Customer Age": 32,
            "Customer Location": "North Melaniehaven",
            "Device Used": "tablet",
            "IP Address": "119.11.178.157",
            "Shipping Address": "7201 Moss Parks Suite 017 Allisonhaven, UT 35559",
            "Billing Address": "7201 Moss Parks Suite 017 Allisonhaven, UT 35559",
            "Account Age Days": 64,
            "Transaction Hour": 19,
            "expected": 0
        },
        {
            "Transaction ID": "03ac285d-4491-4474-8ea8-f577d088cdd6",
            "Customer ID": "c370c8ea-aca1-4648-805d-0a2e4e120497",
            "Transaction Amount": 166.64,
            "Transaction Date": "2024-04-02 23:54:00",
            "Payment Method": "debit card",
            "Product Category": "electronics",
            "Quantity": 3,
            "Customer Age": 30,
            "Customer Location": "Murphyberg",
            "Device Used": "tablet",
            "IP Address": "43.128.96.21",
            "Shipping Address": "6245 Jason Stravenue Blakehaven, NC 22523",
            "Billing Address": "6245 Jason Stravenue Blakehaven, NC 22523",
            "Account Age Days": 71,
            "Transaction Hour": 23,
            "expected": 0
        },
        {
            "Transaction ID": "43d1544c-5730-4c40-bb19-3dd36584d668",
            "Customer ID": "4e2a432b-44e5-4608-9eda-3923604a1163",
            "Transaction Amount": 36.71,
            "Transaction Date": "2024-01-11 06:02:41",
            "Payment Method": "debit card",
            "Product Category": "toys & games",
            "Quantity": 2,
            "Customer Age": 38,
            "Customer Location": "New Megan",
            "Device Used": "mobile",
            "IP Address": "177.0.34.68",
            "Shipping Address": "2839 Santos Walk Richardview, NV 45509",
            "Billing Address": "2839 Santos Walk Richardview, NV 45509",
            "Account Age Days": 182,
            "Transaction Hour": 6,
            "expected": 0
        },
        {
            "Transaction ID": "3c7dc819-4eb0-4416-b834-394df761f29d",
            "Customer ID": "8b785437-dd02-4a2a-9d99-da2da1a08cc5",
            "Transaction Amount": 377.57,
            "Transaction Date": "2024-04-04 23:50:52",
            "Payment Method": "debit card",
            "Product Category": "health & beauty",
            "Quantity": 1,
            "Customer Age": 27,
            "Customer Location": "Phillipsland",
            "Device Used": "desktop",
            "IP Address": "96.7.107.203",
            "Shipping Address": "Unit 9727 Box 3384 DPO AE 79772",
            "Billing Address": "Unit 9727 Box 3384 DPO AE 79772",
            "Account Age Days": 301,
            "Transaction Hour": 23,
            "expected": 0
        },
        {
            "Transaction ID": "4559a360-cf4b-4bac-9edf-bb0b576f6a63",
            "Customer ID": "1b13f66b-d4fb-4092-95d5-2d3df28150a2",
            "Transaction Amount": 101.53,
            "Transaction Date": "2024-03-05 19:35:22",
            "Payment Method": "bank transfer",
            "Product Category": "health & beauty",
            "Quantity": 2,
            "Customer Age": 30,
            "Customer Location": "Rebeccabury",
            "Device Used": "tablet",
            "IP Address": "132.0.11.70",
            "Shipping Address": "878 Waters Points Apt. 497 Lake Louis, PW 57717",
            "Billing Address": "878 Waters Points Apt. 497 Lake Louis, PW 57717",
            "Account Age Days": 4,
            "Transaction Hour": 19,
            "expected": 0
        },
        {
            "Transaction ID": "cf457892-6ea5-4159-bacb-8ae22ff0402f",
            "Customer ID": "23dfaf92-5e80-4719-bc34-7d73405d3b8a",
            "Transaction Amount": 106.98,
            "Transaction Date": "2024-01-13 12:52:06",
            "Payment Method": "credit card",
            "Product Category": "home & garden",
            "Quantity": 5,
            "Customer Age": 30,
            "Customer Location": "Dennisberg",
            "Device Used": "mobile",
            "IP Address": "49.26.206.30",
            "Shipping Address": "251 Robertson Terrace Apt. 144 Devonview, FL 41750",
            "Billing Address": "251 Robertson Terrace Apt. 144 Devonview, FL 41750",
            "Account Age Days": 286,
            "Transaction Hour": 12,
            "expected": 0
        },
        {
            "Transaction ID": "e77d1298-f4db-4e1e-a63c-3450327dd710",
            "Customer ID": "315c7228-1ed5-48da-a42c-f8516df966e8",
            "Transaction Amount": 304.29,
            "Transaction Date": "2024-01-22 18:02:18",
            "Payment Method": "bank transfer",
            "Product Category": "clothing",
            "Quantity": 2,
            "Customer Age": 29,
            "Customer Location": "West Charlesport",
            "Device Used": "desktop",
            "IP Address": "171.189.101.91",
            "Shipping Address": "Unit 6649 Box 9824 DPO AA 27835",
            "Billing Address": "Unit 6649 Box 9824 DPO AA 27835",
            "Account Age Days": 346,
            "Transaction Hour": 18,
            "expected": 0
        },
        {
        "Transaction ID": "c5d2c9cf-3dc1-4977-914a-4ce56e1fdf9a",
        "Customer ID": "e4117527-8629-4787-ac91-d08747bc2471",
        "Transaction Amount": 161.06,
        "Transaction Date": "2024-01-16 13:34",
        "Payment Method": "debit card",
        "Product Category": "toys & games",
        "Quantity": 3,
        "Customer Age": 37,
        "Customer Location": "Jacksonside",
        "Device Used": "tablet",
        "IP Address": "68.86.162.147",
        "Shipping Address": "47718 Wolfe Valleys\nDenisechester, OH 09590",
        "Billing Address": "47718 Wolfe Valleys\nDenisechester, OH 09590",
        "Account Age Days": 249,
        "Transaction Hour": 1,
        "expected": 1
        },   
    ]
    
    # Create payload with all transactions
    transactions_to_send = []
    expected_map = {}
    
    for item in test_data:
        tx_id = item["Transaction ID"][:8]  # Short ID for display
        expected_label = item.pop("expected")
        
        transactions_to_send.append(item)
        expected_map[tx_id] = expected_label
    
    payload = {
        "transactions": transactions_to_send,
        "threshold": 0.5
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            results = response.json()["predictions"]
            
            correct_count = 0
            total = len(results)
            
            for pred in results:
                tx_id = pred["transaction_id"][:8]
                actual = expected_map[tx_id]
                predicted = 1 if pred["is_fraud"] else 0
                prob = pred["fraud_probability"]
                
                status = "✅ MATCH" if actual == predicted else "❌ MISMATCH"
                if actual == predicted:
                    correct_count += 1
                
                print(f"{tx_id:<15} | {actual:<10} | {predicted:<10} | {prob:<12.4f} | {status}")
            
            print("-" * 90)
            print(f"\nResults:")
            print(f"  Correct: {correct_count}/{total}")
            print(f"  Accuracy: {(correct_count/total)*100:.1f}%")
            print(f"  Fraud cases: {sum(expected_map.values())} (expected {list(expected_map.values()).count(1)})")
            
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Failed to connect to {API_URL}")
        print("   Make sure the Flask server is running!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_batch_complete()