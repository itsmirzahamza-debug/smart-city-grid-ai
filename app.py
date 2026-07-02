import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# --- 1. RE-DEFINE THE MODEL ARCHITECTURE ---
class SmartGridLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super(SmartGridLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- 2. LOAD THE EXPORTED BRAIN ENGINES ---
@st.cache_resource
def load_ai_assets():
    model = SmartGridLSTM()
    model.load_state_dict(torch.load('smart_grid_lstm.pth'))
    model.eval()
    scaler = joblib.load('grid_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_ai_assets()
except Exception as e:
    st.error("Could not find 'smart_grid_lstm.pth' or 'grid_scaler.pkl'. Please ensure you exported them from your notebook first!")

# --- 3. FRONTEND UI DESIGN ---
st.set_page_config(page_title="Smart City Grid Control", layout="wide")

st.title("🏙️ AI-Driven Smart City Grid Control Center")
st.markdown("### *Intelligent Demand-Response & Real-Time Virtual Power Plant Optimization*")
st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚡ Live Climate & Load Sensors")
    st.write("Simulate current real-time environmental metrics for the city:")
    
    input_temp = st.slider("Current Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0, step=0.5)
    input_hum = st.slider("Current Humidity (%)", min_value=10, max_value=100, value=55)
    input_dem = st.number_input("Current Grid Load Baseline (MW)", min_value=10.0, max_value=100.0, value=45.0)
    
    st.info("💡 Tip: Set temperature high (e.g. > 40°C) to simulate a heatwave crisis!")

with col2:
    st.header("🤖 Deep Learning Multi-Step Prediction")
    
    if st.button("Execute Grid System Run", type="primary"):
        
        # A. Create sequential data window [1, 24, 3] with user inputs at the final index
        base_sequence = np.zeros((24, 3))
        base_sequence[-1] = [input_temp, input_hum, input_dem]
        
        # B. Apply training scaler parameters
        scaled_seq = scaler.transform(base_sequence)
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
        
        # C. Model Inference Execution
        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()
            
        # D. Mathematically Exact Inverse Scale to Megawatts
        dummy_row = np.zeros((1, 3))
        dummy_row[0, 2] = prediction_scaled  
        
        unscaled_row = scaler.inverse_transform(dummy_row)
        predicted_mw = unscaled_row[0, 2]    
        
        # E. Display Metric Results
        st.metric(label="Predicted Energy Demand (Next Hour Line)", value=f"{predicted_mw:.2f} MW")
        
        # --- 4. EXECUTE REAL-TIME AUTOMATION DECISIONS ---
        st.subheader("📋 System Automation Command Logs")
        GRID_CAPACITY = 65.0
        
        if predicted_mw > GRID_CAPACITY:
            deficit = predicted_mw - GRID_CAPACITY
            st.error(f"🚨 CRITICAL OVERLOAD DETECTED: Forecasted demand exceeds threshold by {deficit:.2f} MW!")
            
            with st.expander("👉 Automated Mitigation Actions Deployed", expanded=True):
                st.code("""
[SYSTEM LOG]: Status 409 - Grid Strain Alert Initiated.
[ACTION 01]: Triggering Protocol Alpha -> Transmitting throttling signals to public EV charging ports.
[ACTION 02]: Triggering Protocol Beta -> Activating 15MW battery storage units injection loop.
[ACTION 03]: Triggering Protocol Gamma -> Dispatched localized app warning notification: 'Peak Demand Spike Active.'
                """, language="bash")
                
        elif predicted_mw < 40.0:
            surplus = GRID_CAPACITY - predicted_mw
            st.success(f"☀️ ECO SURPLUS DETECTED: Forecasted usage drop creates {surplus:.2f} MW cleaner headroom.")
            
            with st.expander("👉 Automated Conservation Actions Deployed", expanded=True):
                st.code("""
[SYSTEM LOG]: Status 202 - Renewable Generation Peak Active.
[ACTION 01]: Triggering Protocol Delta -> Rerouting surplus energy into municipal pumped-hydro wells.
[ACTION 02]: Triggering Protocol Epsilon -> Updating Dynamic Billing API -> Dropping current user unit rate by 40%.
                """, language="bash")
                
        else:
            st.info(f"✅ GRID STABLE: Forecasted load of {predicted_mw:.2f} MW resides inside normal limits.")
            st.write("Grid operations maintaining structural baseline distribution. No intervention overrides mandatory.")