import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from io import BytesIO
from scapy.all import rdpcap, IP, TCP, UDP
import numpy as np 

st.set_page_config(
    page_title="Tor Traffic Classifier",
    page_icon="",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_tor_model.pkl"


@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        return artifact
    except FileNotFoundError:
        st.error("Error: Model file 'best_tor_model.pkl' not found. Please ensure you have run tor_traffic_training.py successfully.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

artifact = load_model()
model = artifact["model"]
feature_columns = artifact["feature_columns"]

st.sidebar.header("Model Info")
st.sidebar.write(f"**Best Model:** {artifact['model_name']}")
st.sidebar.write(f"Number of features: {len(feature_columns)}")

mode = st.sidebar.radio("Select Mode", ["Manual Input", "Upload PCAP"])

def build_flows_from_pcap(pcap_bytes: bytes, max_flows: int = 50):
    
    tmp = BytesIO(pcap_bytes)
    packets = rdpcap(tmp)

    # Flow key: (src_ip, src_port, dst_ip, dst_port, proto)
    flows = {}
    
    # Store the initial flow direction for the first packet of a flow
    flow_init_key = {}

    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        proto = ip.proto
        src_ip = ip.src
        dst_ip = ip.dst

        sport = None
        dport = None
        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        else:
            continue

        key_tuple = tuple(sorted([(src_ip, sport), (dst_ip, dport)])) + (proto,)
        
        t = float(pkt.time)
        size = len(pkt)
        
        # Determine the initial direction and set the canonical key
        if key_tuple not in flows:
            flows[key_tuple] = {
                "times": [],
                "sizes": [],
                "fwd_times": [],
                "bwd_times": [],
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "sport": sport,
                "dport": dport,
                "proto": proto,
            }
            flow_init_key[key_tuple] = (src_ip, sport, dst_ip, dport)
        
        flow = flows[key_tuple]
        flow["times"].append(t)
        flow["sizes"].append(size)
        
        init_src_ip = flow_init_key[key_tuple][0]
        
        if ip.src == init_src_ip:
            flow["fwd_times"].append(t)
        else:
            flow["bwd_times"].append(t)

    feature_rows = []
    # All IAT/Active/Idle features are calculated in seconds, but the model expects microseconds.
    # We must multiply them by 1e6 for consistency with Flow Duration.
    IAT_SCALE = 1e6
    
    for key, flow in list(flows.items())[:max_flows]:
        times = sorted(flow["times"])
        if len(times) < 2:
            continue

        first_t = times[0]
        last_t = times[-1]
        duration = max(last_t - first_t, 1e-6)  # seconds

        total_pkts = len(times)
        total_bytes = sum(flow["sizes"])

        iats = [times[i+1] - times[i] for i in range(len(times)-1)]
        flow_iat_mean = float(pd.Series(iats).mean()) * IAT_SCALE
        flow_iat_std = float(pd.Series(iats).std(ddof=0)) * IAT_SCALE
        flow_iat_max = float(pd.Series(iats).max()) * IAT_SCALE
        flow_iat_min = float(pd.Series(iats).min()) * IAT_SCALE

        fwd_times = sorted(flow["fwd_times"])
        if len(fwd_times) >= 2:
            fwd_iats = [fwd_times[i+1] - fwd_times[i] for i in range(len(fwd_times)-1)]
            fwd_iat_mean = float(pd.Series(fwd_iats).mean()) * IAT_SCALE
            fwd_iat_std = float(pd.Series(fwd_iats).std(ddof=0)) * IAT_SCALE
            fwd_iat_max = float(pd.Series(fwd_iats).max()) * IAT_SCALE
            fwd_iat_min = float(pd.Series(fwd_iats).min()) * IAT_SCALE
        else:
            fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = 0.0

        bwd_times = sorted(flow["bwd_times"])
        if len(bwd_times) >= 2:
            bwd_iats = [bwd_times[i+1] - bwd_times[i] for i in range(len(bwd_times)-1)]
            bwd_iat_mean = float(pd.Series(bwd_iats).mean()) * IAT_SCALE
            bwd_iat_std = float(pd.Series(bwd_iats).std(ddof=0)) * IAT_SCALE
            bwd_iat_max = float(pd.Series(bwd_iats).max()) * IAT_SCALE
            bwd_iat_min = float(pd.Series(bwd_iats).min()) * IAT_SCALE
        else:
            bwd_iat_mean = bwd_iat_std = bwd_iat_max = bwd_iat_min = 0.0

        # Active/Idle approximation
        # Use a threshold: if IAT > 1s, treat as idle gap, otherwise active
        ACTIVE_IDLE_THRESHOLD = 1.0
        active_periods = []
        idle_periods = []

        current_active_start = times[0]
        for i, dt in enumerate(iats):
            if dt <= ACTIVE_IDLE_THRESHOLD:
                continue
            else:
                active_periods.append(times[i] - current_active_start)
                idle_periods.append(dt)
                current_active_start = times[i+1]
        active_periods.append(times[-1] - current_active_start)

        if len(active_periods) > 0:
            active_mean = float(pd.Series(active_periods).mean()) * IAT_SCALE
            active_std = float(pd.Series(active_periods).std(ddof=0)) * IAT_SCALE
            active_max = float(pd.Series(active_periods).max()) * IAT_SCALE
            active_min = float(pd.Series(active_periods).min()) * IAT_SCALE
        else:
            active_mean = active_std = active_max = active_min = 0.0

        if len(idle_periods) > 0:
            idle_mean = float(pd.Series(idle_periods).mean()) * IAT_SCALE
            idle_std = float(pd.Series(idle_periods).std(ddof=0)) * IAT_SCALE
            idle_max = float(pd.Series(idle_periods).max()) * IAT_SCALE
            idle_min = float(pd.Series(idle_periods).min()) * IAT_SCALE
        else:
            idle_mean = idle_std = idle_max = idle_min = 0.0

        row = {
            "Source IP": flow["src_ip"],
            " Source Port": flow["sport"],
            " Destination IP": flow["dst_ip"],
            " Destination Port": flow["dport"],
            " Protocol": flow["proto"],
            " Flow Duration": duration * IAT_SCALE,       
            " Flow Bytes/s": total_bytes / duration,
            " Flow Packets/s": total_pkts / duration,
            " Flow IAT Mean": flow_iat_mean,
            " Flow IAT Std": flow_iat_std,
            " Flow IAT Max": flow_iat_max,
            " Flow IAT Min": flow_iat_min,
            "Fwd IAT Mean": fwd_iat_mean,
            " Fwd IAT Std": fwd_iat_std,
            " Fwd IAT Max": fwd_iat_max,
            " Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Mean": bwd_iat_mean,
            " Bwd IAT Std": bwd_iat_std,
            " Bwd IAT Max": bwd_iat_max,
            " Bwd IAT Min": bwd_iat_min,
            "Active Mean": active_mean,
            " Active Std": active_std,
            " Active Max": active_max,
            " Active Min": active_min,
            "Idle Mean": idle_mean,
            " Idle Std": idle_std,
            " Idle Max": idle_max,
            " Idle Min": idle_min,
        }
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


if mode == "Manual Input":
    st.title("Tor Traffic Classifier – Manual Input Mode")

    DEFAULT_FEATURES = [
        ' Source Port',
        ' Destination Port',
        ' Protocol',
        ' Flow Duration',
        ' Flow Bytes/s',
        ' Flow Packets/s',
        ' Flow IAT Mean',
        ' Flow IAT Std',
        ' Flow IAT Max',
        ' Flow IAT Min',
        'Fwd IAT Mean',
        ' Fwd IAT Std',
        ' Fwd IAT Max',
        ' Fwd IAT Min',
        'Bwd IAT Mean',
        ' Bwd IAT Std',
        ' Bwd IAT Max',
        ' Bwd IAT Min',
        'Active Mean',
        ' Active Std',
        ' Active Max',
        ' Active Min',
        'Idle Mean',
        ' Idle Std',
        ' Idle Max',
        ' Idle Min'
    ]
    
    # Only show the initial 12 features for the manual demo for simplicity
    DISPLAY_FEATURES = DEFAULT_FEATURES[:12]

    missing = [f for f in feature_columns if f not in DEFAULT_FEATURES]
    if missing:
        st.warning(f"Trained model features are missing from the DEFAULT_FEATURES list: {missing}")

    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(DISPLAY_FEATURES):
        default_val = 12000000.0 if 'Duration' in feat or 'IAT' in feat or 'Active' in feat or 'Idle' in feat else 1000.0
        user_input[feat] = cols[i % 3].number_input(feat, value=default_val)
    
    st.caption("Note: Only the first 12 features are shown, but the model requires all 26. Missing features are set to 0.0.")


    if st.button("Predict (Manual Input)"):
        data_dict = {col: 0.0 for col in feature_columns}
        for feat in DISPLAY_FEATURES:
            if feat in data_dict:
                 data_dict[feat] = user_input[feat]

        input_df = pd.DataFrame([data_dict])
        
        try:
            pred = model.predict(input_df)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]

            if pred == 1:
                st.success("Prediction: **Tor Traffic Detected**")
            else:
                st.info("Prediction: **Non-Tor / Normal Traffic**")

            if prob is not None:
                st.write(f"Estimated probability of Tor traffic: **{prob*100:.2f}%**")
        except Exception as e:
            st.error(f"Prediction Error: {e}")


else:
    st.title("Tor Traffic Classification – PCAP Mode")

    st.write("""
    Upload a `.pcap` or `.pcapng` file. The app will:
    1. Parse packets using Scapy and compute **26 flow features (in µs)**.  
    2. Run the ML model on **all** extracted flows.  
    3. **Aggregate:** If **any** flow is classified as Tor, the overall result is **Tor Traffic Detected**.
    """)

    uploaded_pcap = st.file_uploader("Upload PCAP file", type=["pcap", "pcapng"])

    if uploaded_pcap is not None:
        with st.spinner("Extracting flows and computing features from PCAP..."):
            flow_df = build_flows_from_pcap(uploaded_pcap.read(), max_flows=50)

        if flow_df.empty:
            st.warning("No valid TCP/UDP flows found in this PCAP.")
        else:
            st.success(f"Successfully extracted {len(flow_df)} flows from PCAP.")
            
            st.write("### Aggregated PCAP Prediction")
            
            predict_df_list = []
            
            for _, flow in flow_df.iterrows():
                data_dict = {col: 0.0 for col in feature_columns}
                for col in flow_df.columns:
                    if col in data_dict:
                        data_dict[col] = flow[col]
                predict_df_list.append(data_dict)
            
            prediction_input_df = pd.DataFrame(predict_df_list)
            
            try:
                predictions = model.predict(prediction_input_df)
                
                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(prediction_input_df)[:, 1]

                tor_flows_count = (predictions == 1).sum()
                
                if tor_flows_count > 0:
                    overall_pred = 1
                    st.success(f"**Overall Prediction: Tor Traffic Detected** (Found {tor_flows_count} of {len(flow_df)} flows classified as Tor)")
                else:
                    overall_pred = 0
                    st.info("**Overall Prediction: Non-Tor / Normal Traffic**")

                if probabilities is not None:
                    max_prob = probabilities.max() if len(probabilities) > 0 else 0.0
                    st.write(f"Highest estimated probability of Tor traffic across all flows: **{max_prob*100:.2f}%**")

                flow_df["Predicted Label"] = ["Tor" if p == 1 else "Non-Tor" for p in predictions]
                if probabilities is not None:
                    flow_df["Tor Probability"] = [f"{p*100:.2f}%" for p in probabilities]

                st.write("### Flow-by-Flow Results")
                display_cols = ["Source IP", " Source Port", " Destination IP",
                                " Destination Port", " Protocol", " Flow Duration", 
                                "Predicted Label", "Tor Probability"]
                 
                st.dataframe(flow_df[[c for c in display_cols if c in flow_df.columns]])
            
            except Exception as e:
                st.error(f"Prediction Error: {e}")