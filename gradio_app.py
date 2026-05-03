import gradio as gr
import requests
import pandas as pd
import json

# ── Config — change this to your Cloud Run URL when deployed ─────────────────
API_BASE_URL = "http://127.0.0.1:8000"  # change to your Cloud Run URL

MODELS = ["logistic_regression", "XGBOOST", "random_forest"]

# ── Predict from manual input ─────────────────────────────────────────────────
def predict_manual(
    model_name,
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies, paperless_billing,
    monthly_charges, total_charges,
    multiplelines_no_phone, multiplelines_yes,
    internet_fiber, internet_no,
    contract_one_year, contract_two_year,
    payment_credit_card, payment_electronic, payment_mailed
):
    payload = {
        "model_name":                        model_name,
        "gender":                            int(gender),
        "SeniorCitizen":                     int(senior_citizen),
        "Partner":                           int(partner),
        "Dependents":                        int(dependents),
        "tenure":                            float(tenure),
        "PhoneService":                      int(phone_service),
        "OnlineSecurity":                    int(online_security),
        "OnlineBackup":                      int(online_backup),
        "DeviceProtection":                  int(device_protection),
        "TechSupport":                       int(tech_support),
        "StreamingTV":                       int(streaming_tv),
        "StreamingMovies":                   int(streaming_movies),
        "PaperlessBilling":                  int(paperless_billing),
        "MonthlyCharges":                    float(monthly_charges),
        "TotalCharges":                      float(total_charges),
        "MultipleLines_No_phone_service":    int(multiplelines_no_phone),
        "MultipleLines_Yes":                 int(multiplelines_yes),
        "InternetService_Fiber_optic":       int(internet_fiber),
        "InternetService_No":                int(internet_no),
        "Contract_One_year":                 int(contract_one_year),
        "Contract_Two_year":                 int(contract_two_year),
        "PaymentMethod_Credit_card_automatic": int(payment_credit_card),
        "PaymentMethod_Electronic_check":    int(payment_electronic),
        "PaymentMethod_Mailed_check":        int(payment_mailed),
    }

    try:
        response = requests.post(f"{API_BASE_URL}/predict/manual", json=payload)
        response.raise_for_status()
        data = response.json()
        pred = data["prediction"]

        churn_label = "⚠️ Will Churn" if pred["churn"] else "✅ Will NOT Churn"
        prob        = pred["churn_probability"]
        risk        = pred["risk_level"]

        return (
            f"{churn_label}",
            f"{prob:.1%}",
            f"{risk} Risk",
            f"Model used: {data['model_used']}"
        )
    except requests.exceptions.ConnectionError:
        return "❌ API not reachable", "", "", "Check if FastAPI server is running"
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


# ── Predict from CSV upload ────────────────────────────────────────────────────
def predict_csv(model_name, csv_file):
    if csv_file is None:
        return "Please upload a CSV file", None

    try:
        with open(csv_file.name, "rb") as f:
            response = requests.post(
                f"{API_BASE_URL}/predict/csv",
                params={"model_name": model_name},
                files={"file": (csv_file.name, f, "text/csv")}
            )
        response.raise_for_status()
        data = response.json()

        predictions = data["predictions"]
        df = pd.DataFrame(predictions)
        df.columns = ["Churn", "Probability", "Risk Level"]
        df["Churn"] = df["Churn"].map({True: "⚠️ Yes", False: "✅ No"})
        df["Probability"] = df["Probability"].apply(lambda x: f"{x:.1%}")

        summary = (
            f"Total rows: {data['total_rows']} | "
            f"Model: {data['model_used']} | "
            f"Churners: {sum(1 for p in predictions if p['churn'])} | "
            f"Non-churners: {sum(1 for p in predictions if not p['churn'])}"
        )
        return summary, df

    except requests.exceptions.ConnectionError:
        return "❌ API not reachable — check if FastAPI server is running", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Churn Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Customer Churn Predictor")
    gr.Markdown("Predict whether a customer will churn using Logistic Regression, XGBoost, or Random Forest.")

    model_dropdown = gr.Dropdown(
        choices=MODELS,
        value="xgboost",
        label="Select Model"
    )

    with gr.Tabs():

        # ── Tab 1: Manual Input ──────────────────────────────────────────────
        with gr.TabItem("Manual Input"):
            gr.Markdown("### Enter Customer Features")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Demographics**")
                    gender          = gr.Radio(choices=[(("Male", 1)), ("Female", 0)], value=1, label="Gender")
                    senior_citizen  = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Senior Citizen")
                    partner         = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Partner")
                    dependents      = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Dependents")
                    tenure          = gr.Slider(minimum=0, maximum=72, value=12, step=1, label="Tenure (months)")

                with gr.Column():
                    gr.Markdown("**Services**")
                    phone_service    = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=1, label="Phone Service")
                    online_security  = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Online Security")
                    online_backup    = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Online Backup")
                    device_protect   = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Device Protection")
                    tech_support     = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Tech Support")
                    streaming_tv     = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Streaming TV")
                    streaming_movies = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Streaming Movies")

                with gr.Column():
                    gr.Markdown("**Billing**")
                    paperless_billing = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=1, label="Paperless Billing")
                    monthly_charges   = gr.Number(value=65.0, label="Monthly Charges ($)")
                    total_charges     = gr.Number(value=780.0, label="Total Charges ($)")

                    gr.Markdown("**Multiple Lines**")
                    multiplelines_no_phone = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="No Phone Service")
                    multiplelines_yes      = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Multiple Lines")

                    gr.Markdown("**Internet Service**")
                    internet_fiber = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Fiber Optic")
                    internet_no    = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="No Internet")

                    gr.Markdown("**Contract**")
                    contract_one_year = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="One Year Contract")
                    contract_two_year = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Two Year Contract")

                    gr.Markdown("**Payment Method**")
                    payment_credit_card = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Credit Card (Auto)")
                    payment_electronic  = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=1, label="Electronic Check")
                    payment_mailed      = gr.Radio(choices=[("Yes", 1), ("No", 0)], value=0, label="Mailed Check")

            predict_btn = gr.Button("🔍 Predict", variant="primary")

            with gr.Row():
                out_churn = gr.Textbox(label="Prediction")
                out_prob  = gr.Textbox(label="Churn Probability")
                out_risk  = gr.Textbox(label="Risk Level")
                out_model = gr.Textbox(label="Model Info")

            predict_btn.click(
                fn=predict_manual,
                inputs=[
                    model_dropdown,
                    gender, senior_citizen, partner, dependents, tenure,
                    phone_service, online_security, online_backup, device_protect,
                    tech_support, streaming_tv, streaming_movies, paperless_billing,
                    monthly_charges, total_charges,
                    multiplelines_no_phone, multiplelines_yes,
                    internet_fiber, internet_no,
                    contract_one_year, contract_two_year,
                    payment_credit_card, payment_electronic, payment_mailed
                ],
                outputs=[out_churn, out_prob, out_risk, out_model]
            )

        # ── Tab 2: CSV Upload ────────────────────────────────────────────────
        with gr.TabItem("CSV Upload"):
            gr.Markdown("### Upload a CSV file with customer data")
            gr.Markdown(
                "CSV must have these columns (already preprocessed — binary encoded, OHE applied):\n"
                "`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, "
                "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, "
                "StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges, "
                "MultipleLines_No phone service, MultipleLines_Yes, InternetService_Fiber optic, "
                "InternetService_No, Contract_One year, Contract_Two year, "
                "PaymentMethod_Credit card (automatic), PaymentMethod_Electronic check, "
                "PaymentMethod_Mailed check`"
            )

            csv_file    = gr.File(label="Upload CSV", file_types=[".csv"])
            csv_btn     = gr.Button("🔍 Predict All Rows", variant="primary")
            csv_summary = gr.Textbox(label="Summary")
            csv_table   = gr.Dataframe(label="Predictions")

            csv_btn.click(
                fn=predict_csv,
                inputs=[model_dropdown, csv_file],
                outputs=[csv_summary, csv_table]
            )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)

demo.launch(share=True)
