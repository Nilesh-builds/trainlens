"""Generate realistic customer support conversation dataset."""
import csv
import random
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

CATEGORIES = ["billing", "technical_support", "shipping", "product_inquiry", "cancellation", "refund"]

INTENTS = {
    "billing": ["charge_question", "payment_failed", "invoice_request", "plan_upgrade", "prorated_charge"],
    "technical_support": ["app_crash", "login_issue", "feature_not_working", "slow_performance", "error_message"],
    "shipping": ["track_order", "delayed_delivery", "wrong_address", "missing_package", "return_label"],
    "product_inquiry": ["feature_request", "compatibility", "pricing", "availability", "comparison"],
    "cancellation": ["cancel_subscription", "downgrade_plan", "pause_account", "delete_account"],
    "refund": ["full_refund", "partial_refund", "refund_status", "refund_timeline", "wrong_item_refund"],
}

CHANNELS = ["email", "chat", "phone", "social_media"]
SENTIMENTS = ["positive", "neutral", "negative"]
PRIORITIES = ["low", "medium", "high", "critical"]

CUSTOMER_MSGS = {
    "charge_question": [
        "I was charged twice for my subscription this month",
        "Why is there an extra $4.99 on my credit card statement?",
        "I see a charge I don't recognize from your company",
        "My invoice shows a different amount than what I agreed to",
    ],
    "payment_failed": [
        "My payment keeps failing even though I have sufficient funds",
        "I tried updating my card but it says declined",
        "The system won't accept my new credit card number",
    ],
    "invoice_request": [
        "Can you send me an invoice for last month's charges?",
        "I need a tax invoice for my business expenses",
        "Where can I download my past invoices?",
    ],
    "plan_upgrade": [
        "I want to upgrade from basic to premium plan",
        "How much does the enterprise plan cost?",
        "Can I switch to annual billing to save money?",
    ],
    "prorated_charge": [
        "Will I get charged extra if I upgrade mid-cycle?",
        "How is the prorated amount calculated?",
    ],
    "app_crash": [
        "The app crashes every time I open it on Android",
        "Your application keeps force closing on my phone",
        "I can't even use the app because it keeps crashing",
    ],
    "login_issue": [
        "I can't log into my account",
        "The password reset email never arrives",
        "My account is locked after too many attempts",
    ],
    "feature_not_working": [
        "The search function returns no results",
        "Notifications are not coming through anymore",
        "The export feature gives me an error",
    ],
    "slow_performance": [
        "The website is incredibly slow today",
        "Pages take forever to load",
        "Your service has been sluggish for the past week",
    ],
    "error_message": [
        "I keep getting error code 500 when trying to save",
        "There's an error saying 'something went wrong'",
        "I get a blank white screen when I try to checkout",
    ],
    "track_order": [
        "Where is my order? It's been 5 days",
        "Can you give me the tracking number for order #12345?",
        "My package hasn't moved in 3 days according to tracking",
    ],
    "delayed_delivery": [
        "My order was supposed to arrive yesterday but nothing came",
        "The delivery is 2 weeks late now",
        "FedEx says delivered but I never received it",
    ],
    "wrong_address": [
        "I need to change the shipping address before it ships",
        "I put the wrong apartment number on my order",
        "Can you update the delivery address for my pending order?",
    ],
    "missing_package": [
        "Tracking says delivered but my package is not here",
        "Someone might have stolen my package",
        "The courier left it at the wrong door",
    ],
    "return_label": [
        "How do I get a return label?",
        "I need to return this item, what are the steps?",
        "Can you email me a prepaid return label?",
    ],
    "feature_request": [
        "It would be great if you added dark mode",
        "Can you add a bulk export option?",
        "Please consider adding integration with Slack",
    ],
    "compatibility": [
        "Does this work with Windows 11?",
        "Is your product compatible with macOS Sonoma?",
        "Will this work on my old iPad?",
    ],
    "pricing": [
        "What are the pricing tiers?",
        "Do you offer student discounts?",
        "Is there a free trial available?",
    ],
    "availability": [
        "When will the blue variant be back in stock?",
        "Is this item available in India?",
        "When is the new model releasing?",
    ],
    "comparison": [
        "What's the difference between basic and pro?",
        "How does your product compare to Competitor X?",
        "Which plan includes API access?",
    ],
    "cancel_subscription": [
        "I want to cancel my subscription immediately",
        "How do I cancel my account?",
        "Please cancel my membership",
    ],
    "downgrade_plan": [
        "Can I downgrade to the free tier?",
        "I want to switch to a cheaper plan",
        "Is there a way to reduce my monthly cost?",
    ],
    "pause_account": [
        "Can I pause my subscription for a few months?",
        "Is there an option to temporarily suspend my account?",
    ],
    "delete_account": [
        "I want to permanently delete my account and all data",
        "How do I remove my account completely?",
        "Please delete all my information from your system",
    ],
    "full_refund": [
        "I want a full refund for this order",
        "This product doesn't work, I need my money back",
        "I was charged incorrectly, I want the full amount back",
    ],
    "partial_refund": [
        "Can I get a partial refund since the item was damaged?",
        "I'd like a refund for the shipping cost only",
        "One item was missing, can I get a partial refund?",
    ],
    "refund_status": [
        "When will my refund be processed?",
        "I still haven't received my refund from 2 weeks ago",
        "How long do refunds usually take?",
    ],
    "refund_timeline": [
        "How many days does a refund take?",
        "Can you tell me when to expect the refund?",
        "Is there a specific date my refund will arrive?",
    ],
    "wrong_item_refund": [
        "I received the wrong item, I need a refund and the correct one",
        "The product I got is completely different from what I ordered",
    ],
}

AGENT_MSGS = [
    "I understand your concern. Let me look into this for you right away.",
    "Thank you for reaching out. I can definitely help you with that.",
    "I apologize for the inconvenience. Let me resolve this for you.",
    "That's a great question. Let me pull up the information for you.",
    "I completely understand your frustration. Here's what I can do.",
    "Let me check our system and get back to you with the details.",
    "I'm sorry to hear that. Let me escalate this to our specialist team.",
    "I can see the issue in our system. Let me fix it now.",
    "According to our records, here's what happened...",
    "I've initiated the process for you. You should see it reflected within 24 hours.",
]

RESOLUTIONS = {
    "charge_question": "investigated_charge",
    "payment_failed": "updated_payment_method",
    "invoice_request": "sent_invoice",
    "plan_upgrade": "upgraded_plan",
    "prorated_charge": "explained_billing",
    "app_crash": "provided_fix",
    "login_issue": "reset_account",
    "feature_not_working": "escalated_to_tech",
    "slow_performance": "reported_to_engineering",
    "error_message": "provided_workaround",
    "track_order": "provided_tracking",
    "delayed_delivery": "escalated_to_logistics",
    "wrong_address": "updated_address",
    "missing_package": "initiated_investigation",
    "return_label": "sent_return_label",
    "feature_request": "logged_feedback",
    "compatibility": "confirmed_compatibility",
    "pricing": "provided_pricing_info",
    "availability": "checked_inventory",
    "comparison": "provided_comparison",
    "cancel_subscription": "processed_cancellation",
    "downgrade_plan": "processed_downgrade",
    "pause_account": "paused_account",
    "delete_account": "initiated_deletion",
    "full_refund": "processed_full_refund",
    "partial_refund": "processed_partial_refund",
    "refund_status": "provided_refund_status",
    "refund_timeline": "provided_timeline",
    "wrong_item_refund": "processed_refund_and_reship",
}

CSAT_SCORES = [1, 2, 3, 4, 5]
RESPONSE_TIMES = list(range(30, 3600))


def generate_conversation(category: str, intent: str, conv_id: int) -> dict:
    """Generate a single conversation record."""
    customer_msgs = CUSTOMER_MSGS[intent]
    customer_msg = random.choice(customer_msgs)
    agent_msg = random.choice(AGENT_MSGS)
    resolution = RESOLUTIONS[intent]

    num_turns = random.randint(2, 6)
    conversation_history = []
    for turn in range(num_turns):
        if turn % 2 == 0:
            conversation_history.append({"role": "customer", "content": random.choice(customer_msgs)})
        else:
            conversation_history.append({"role": "agent", "content": random.choice(AGENT_MSGS)})

    sentiment = random.choices(SENTIMENTS, weights=[0.3, 0.4, 0.3])[0]
    if resolution in ("processed_full_refund", "processed_cancellation"):
        sentiment = random.choices(SENTIMENTS, weights=[0.1, 0.3, 0.6])[0]
    elif resolution in ("provided_tracking", "confirmed_compatibility", "provided_pricing_info"):
        sentiment = random.choices(SENTIMENTS, weights=[0.6, 0.3, 0.1])[0]

    priority = random.choices(PRIORITIES, weights=[0.2, 0.5, 0.2, 0.1])[0]
    if category in ("refund", "cancellation"):
        priority = random.choices(PRIORITIES, weights=[0.1, 0.3, 0.4, 0.2])[0]

    base_date = datetime(2024, 1, 1)
    created_at = base_date + timedelta(days=random.randint(0, 364), hours=random.randint(8, 22), minutes=random.randint(0, 59))
    response_time = random.choice(RESPONSE_TIMES)

    customer_satisfaction = random.choices(CSAT_SCORES, weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0]
    if sentiment == "negative":
        customer_satisfaction = random.choices(CSAT_SCORES, weights=[0.4, 0.3, 0.2, 0.08, 0.02])[0]
    elif sentiment == "positive":
        customer_satisfaction = random.choices(CSAT_SCORES, weights=[0.02, 0.05, 0.13, 0.35, 0.45])[0]

    return {
        "conversation_id": f"CONV-{conv_id:05d}",
        "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "channel": random.choice(CHANNELS),
        "category": category,
        "intent": intent,
        "priority": priority,
        "customer_message": customer_msg,
        "agent_response": agent_msg,
        "conversation_turns": num_turns,
        "conversation_history": str(conversation_history),
        "resolution": resolution,
        "sentiment": sentiment,
        "customer_satisfaction": customer_satisfaction,
        "response_time_seconds": response_time,
        "is_resolved": random.choices([True, False], weights=[0.85, 0.15])[0],
        "agent_id": f"AGT-{random.randint(1001, 1050):04d}",
    }


def inject_quality_issues(records: list[dict]) -> list[dict]:
    """Inject realistic data quality issues."""
    for i, record in enumerate(records):
        if random.random() < 0.03:
            record["customer_message"] = ""
        if random.random() < 0.02:
            record["category"] = ""
        if random.random() < 0.015:
            record["sentiment"] = random.choice(["happy", "sad", "angry"])
        if random.random() < 0.01:
            record["customer_satisfaction"] = 99
        if random.random() < 0.02:
            record["channel"] = "unknown_channel"
        if random.random() < 0.01:
            record["created_at"] = "invalid-date"
        if random.random() < 0.005:
            duplicate = record.copy()
            duplicate["conversation_id"] = record["conversation_id"]
            records.insert(i + 1, duplicate)
    return records


def main():
    records = []
    conv_id = 1
    for category, intents in INTENTS.items():
        for intent in intents:
            n_samples = random.randint(15, 40)
            for _ in range(n_samples):
                records.append(generate_conversation(category, intent, conv_id))
                conv_id += 1

    random.shuffle(records)
    records = inject_quality_issues(records)

    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "customer_support_tickets.csv"

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} conversations -> {output_path}")


if __name__ == "__main__":
    main()
