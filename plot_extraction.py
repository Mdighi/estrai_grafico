import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import io
import zipfile
import tempfile
import warnings
import pickle
import joblib
import openpyxl
import sympy
import xlsxwriter
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

from scipy.spatial.distance import (
    cdist,
    euclidean,
    cosine,
    cityblock
)
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit, minimize
from scipy.stats import zscore, f_oneway
from scipy.ndimage import binary_dilation
from scipy.signal import medfilt

from skimage.measure import find_contours
from io import BytesIO
st.set_page_config(layout="wide")
def run():
   




    st.title("WORKSPACE")

    tab_labels = [
        "<span style='font-size: 40px; color: #6BE88D;'>**Module A**: Data Generation</span>",
        "<span style='font-size: 40px; color: #EB7323;'>**Module B**: Train Model</span>",
        "<span style='font-size: 40px; color: #4B4BFF;'>**Module C**: Classification</span>"
    ]

    tab1, tab2, tab3 = st.tabs(["Module A", "Module B", "Module C"])
    # --- Inizializzazione completa per i parametri della Sidebar ---
    # I valori qui sono i default che verranno usati al primo avvio o dopo un reset.

    # Editing Immagine
    if "enable_data_filter" not in st.session_state: st.session_state.enable_data_filter = False
    if "invert_threshold" not in st.session_state: st.session_state.invert_threshold = True
    if "filter_window_size" not in st.session_state: st.session_state.filter_window_size = 7
    if "iqr_multiplier" not in st.session_state: st.session_state.iqr_multiplier = 1.5
    if "threshold_val" not in st.session_state: st.session_state.threshold_val = 200
    if "alpha_contrast" not in st.session_state: st.session_state.alpha_contrast = 1.0
    if "beta_brightness" not in st.session_state: st.session_state.beta_brightness = 0.0
    if "tol_dx" not in st.session_state: st.session_state.tol_dx = 10
    if "tol_dy" not in st.session_state: st.session_state.tol_dy = 10
    if "canny_low" not in st.session_state: st.session_state.canny_low = 50
    if "canny_high" not in st.session_state: st.session_state.canny_high = 150
    if "hough_thresh" not in st.session_state: st.session_state.hough_thresh = 100
    if "hough_min_length" not in st.session_state: st.session_state.hough_min_length = 100
    if "hough_max_gap" not in st.session_state: st.session_state.hough_max_gap = 20
    if "mask_curve_only" not in st.session_state: st.session_state.mask_curve_only = None
    # Editing Assi
    if "center_plot" not in st.session_state: st.session_state.center_plot = True
    if "x0_pix" not in st.session_state: st.session_state.x0_pix = 0
    if "y0_pix" not in st.session_state: st.session_state.y0_pix = 0
    if "x1_pix" not in st.session_state: st.session_state.x1_pix = 100
    if "y1_pix" not in st.session_state: st.session_state.y1_pix = 100
    if "x0_val" not in st.session_state: st.session_state.x0_val = 0.0
    if "y0_val" not in st.session_state: st.session_state.y0_val = 0.0
    if "x1_val" not in st.session_state: st.session_state.x1_val = 10.0
    if "y1_val" not in st.session_state: st.session_state.y1_val = 10.0

    # Metodi Matematici
    if "Combine_Methods_Bt" not in st.session_state: st.session_state.Combine_Methods_Bt = False
    if "fit_method_Reg" not in st.session_state: st.session_state.fit_method_Reg = "No_Reg"
    if "fit_method_Clust" not in st.session_state: st.session_state.fit_method_Clust = "No_Clust"
    if "N_color_clusters" not in st.session_state: st.session_state.N_color_clusters = 3
    if "dbscan_min_samples" not in st.session_state: st.session_state.dbscan_min_samples = 15
    if "dbscan_eps" not in st.session_state: st.session_state.dbscan_eps = 20
    if "pixel_proximity_threshold" not in st.session_state: st.session_state.pixel_proximity_threshold = 10
    if "perimeter_offset_radius" not in st.session_state: st.session_state.perimeter_offset_radius = 3
    if "perimeter_smoothing_sigma" not in st.session_state: st.session_state.perimeter_smoothing_sigma = 1.0
    if "path_fit_type" not in st.session_state: st.session_state.path_fit_type = "Spline"
    if "window_size" not in st.session_state: st.session_state.window_size = 5
    if "hidden_layers" not in st.session_state: st.session_state.hidden_layers = "10,10"
    if "activation" not in st.session_state: st.session_state.activation = "tanh"
    if "max_iter" not in st.session_state: st.session_state.max_iter = 10000
    if "forecast_length" not in st.session_state: st.session_state.forecast_length = 2.0
    if "approx_fourier" not in st.session_state: st.session_state.approx_fourier = False
    if "fourier_approx_harmonics" not in st.session_state: st.session_state.fourier_approx_harmonics = 5
    if "feedback_file_names" not in st.session_state:
        st.session_state.feedback_file_names = []
    if "feedback_labels" not in st.session_state:
        st.session_state.feedback_labels = []
    if "feedback_features" not in st.session_state:
        st.session_state.feedback_features = []
    # ... (Il resto del tuo script Streamlit) ...
    # Inizializzazioni delle variabili di sessione che hai fornito
    if "y_approx_fourier" not in st.session_state:
        st.session_state.y_approx_fourier = np.array([])
    if "x_extended" not in st.session_state:
        st.session_state.x_extended = np.array([])
    if "y_fit_primary" not in st.session_state:
        st.session_state.y_fit_primary = np.array([])
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "fit_method_Clust" not in st.session_state:
        st.session_state.fit_method_Clust=None
    # In your initial st.session_state setup at the very top of your script:
    if 'cluster_labels' not in st.session_state:
        st.session_state.cluster_labels = [] # Inizializzato come lista vuota
    if "temp_y_approx_fourier" not in st.session_state:
        st.session_state.temp_y_approx_fourier = np.array([])
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = None
        # ... (then further down, inside train_curve_classifier)
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.5  
        # Inizializzazioni per la session_state (se non già presenti)
    if 'all_feature_names' not in st.session_state:
        st.session_state.all_feature_names = [] 
    if 'processed_dataframes' not in st.session_state:
        st.session_state.processed_dataframes = {}
    if 'reference_curve_name' not in st.session_state:
        st.session_state.reference_curve_name = None
    if 'current_test_curve_index' not in st.session_state:
        st.session_state.current_test_curve_index = 0
    if 'training_feedback' not in st.session_state:
        st.session_state.training_feedback = []
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.5
    if 'model_features_data' not in st.session_state:
        st.session_state.model_features_data = pd.DataFrame(columns=['features', 'label'])
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_report' not in st.session_state:
        st.session_state.model_report = None
    if 'training_feedback_complete' not in st.session_state:
        st.session_state.training_feedback_complete = False
        # --- Definizione della funzione train_curve_classifier ---
    if "manhattan_dist"not in st.session_state:
        st.session_state.manhattan_dist=None
    if "colore_cluster"not in st.session_state:
        st.session_state.colore_cluster=None
    if "exclude_outlier"not in st.session_state:
        st.session_state.exclude_outlier=False
    if "var_outlier_zscore" not in st.session_state or st.session_state["var_outlier_zscore"] in [False, 0, None]:
        st.session_state["var_outlier_zscore"] = 2.5

    def load_and_apply_sidebar_params(uploaded_bytes):
        import io
        df = pd.read_csv(io.BytesIO(uploaded_bytes))

        # Prendi solo le colonne Param_...
        param_cols = [c for c in df.columns if c.startswith("Param_")]
        if not param_cols:
            st.warning("⚠ Nessuna colonna 'Param_' trovata.")
            return

        row = df[param_cols].iloc[0]
        key_map = {
            "Param_Fit_Method_Reg": "fit_method_Reg",
            "Param_Fit_Method_Clust": "fit_method_Clust",
        }
        reg_options = ["No_Reg","Linear","Polinomial grade 2","Movie means",
                    "Spline","Exponential","Fourier","Neural Network",
                    "Gradient Boosting","Support Vector Regression",
                    "Symbolic Regression","Random Forest Regression"]
        clust_options = ["No_Clust","Closed path","Cluster Color","Open Path"]

        for csv_key, val in row.items():
            if pd.isna(val): continue
            key = key_map.get(csv_key, csv_key.replace("Param_", ""))
            if key not in st.session_state: continue

            # MATCH robusto per fit_method_Reg
            if key == "fit_method_Reg":
                matched = next((o for o in reg_options
                                if str(val).strip().lower().replace(" ", "") == o.lower().replace(" ", "")), None)
                st.session_state[key] = matched or reg_options[0]

            # MATCH robusto per fit_method_Clust
            elif key == "fit_method_Clust":
                matched = next((o for o in clust_options
                                if str(val).strip().lower().replace(" ", "") == o.lower().replace(" ", "")), None)
                st.session_state[key] = matched or clust_options[0]

            # gli altri Param_...
            else:
                old = st.session_state[key]
                t = type(old)
                try:
                    if t is bool:   st.session_state[key] = str(val).lower() in ("true","1","t","yes")
                    elif t is int:  st.session_state[key] = int(float(val))
                    elif t is float:st.session_state[key] = float(val)
                    else:           st.session_state[key] = val
                except:
                    st.warning(f"⚠ Non posso convertire {key}={val}")

        st.success("✅ Parametri applicati!")


    mask_curve_centered = None    
        

        # Initialize variables that might be used before file upload
    center_plot = True # Default value
    x0_pix, x1_pix, x0_val, x1_val = 0, 100, 0.0, 10.0
    y0_pix, y1_pix, y0_val, y1_val = 0, 100, 0.0, 10.0
    hidden_layers_default = "8" # Default for NN if not selected
    stroke_width_value=5
        # All'interno di "with tab1:", sostituisci l'intero blocco della sidebar con questo.

    with st.sidebar:        
        st.title("SIDEBAR")
        st.markdown("<h2 style='color: #6BE88D;'>Module A</h2>", unsafe_allow_html=True)
                        # --- SEZIONE PER CARICARE I PARAMETRI ---
            # — Se ci sono parametri in pending, applicali e rerun —
        if st.session_state.get("pending_params_file_bytes"):
            load_and_apply_sidebar_params(st.session_state.pending_params_file_bytes)
            st.session_state.pending_params_file_bytes = None
            st.rerun()

        with st.expander("⚙️Editing Immagine"):
            st.subheader("Data Filter", help="If you want to give the image an initial clean up.")
            st.checkbox("Enable Data Filter", key="enable_data_filter")
            st.checkbox("Invert threshold (THRESH_BINARY_INV)", key="invert_threshold")

            if st.session_state.enable_data_filter:
                st.slider("Filter Window Size (points)", min_value=3, max_value=51, step=2, key="filter_window_size", help="Determines the smoothing level and sensitivity to outliers. Must be an odd number.")
                st.slider("Moltiplicatore IQR per Outlier", min_value=1.0, max_value=5.0, step=0.1, key="iqr_multiplier", help="Lower values remove more outliers, higher values remove fewer..")
            st.subheader("Curve threshold to consider, important parameter")
            st.slider("Threshold value", 0, 400, key="threshold_val")

            st.subheader("Contrast and Brightness Adjustment")
            st.slider(
                "Contrast (Alpha)",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                key="alpha_contrast",
                help="Controls the intensity of image contrast. Values >1 increase, <1 decrease."
            )
            st.slider(
                "Brightness (Beta)",
                min_value=0.0,
                max_value=10.0,
                step=0.05,
                key="beta_brightness",
                help="Controls the image brightness. Positive values increase, negative decrease."
            )

            st.slider("Tolerance dx vertical lines", 0, 50, key="tol_dx")
            st.slider("Tolerance dy horizontal lines", 1, 50, key="tol_dy")
            st.slider("Canny lower threshold", 1, 300, key="canny_low")
            st.slider("Canny upper threshold", 1, 500, key="canny_high")
            st.slider("HoughLinesP threshold", 1, 300, key="hough_thresh")
            st.slider("HoughLinesP minLineLength", 1, 500, key="hough_min_length")
            st.slider("HoughLinesP maxLineGap", 1, 300, key="hough_max_gap")

            # Note: The Canvas logic is for temporary interaction and is not saved/loaded.
            # Therefore we don't use 'key' for these widgets.
            checkbox_canvas = st.checkbox("Enable Manual Editing", value=False)
            st.info(
                "If parameter editing is not enough, you can use a pen and eraser to edit the grayscale image, then view the changes in the *🟢Image for fit* and *Fitting* canvases in the WORKSPACE."
            )
            if checkbox_canvas:
                st.subheader(
                    "🖌️ Canvas",
                    help="Go to the Manual Editing canvas and use the mouse to modify the image, then upload it."
                )
                editing_mode = st.sidebar.radio(
                    "Manual Editing Mode:",
                    ("None_Pen", "Pen (Black)", "Pen (White)"),
                    key="editing_mode_radio"
                )
                stroke_width_value = st.sidebar.slider("Brush Thickness", 1, 20, 5)

        st.markdown("---")

        with st.expander("⚙️ Axis Editing"):
            st.checkbox("Automatically center the plot", key="center_plot")
            st.subheader("Scale")
            st.number_input("X-axis pixel (origin)", key="x0_pix")
            st.number_input("X-axis pixel (second point)", key="x1_pix")
            st.number_input("Real X value at x0", key="x0_val")
            st.number_input("Real X value at x1", key="x1_val")
            st.number_input("Y-axis pixel (origin)", key="y0_pix")
            st.number_input("Y-axis pixel (second point)", key="y1_pix")
            st.number_input("Real Y value at y0", key="y0_val")
            st.number_input("Real Y value at y1", key="y1_val")

        st.markdown("---")

        with st.expander("⚙️ Mathematical Methods"):
            available_methods = [
                "No_Reg", "Linear", "Polynomial degree 2", "Moving means", "Spline",
                "Exponential", "Fourier", "Neural Network", "Gradient Boosting",
                "Support Vector Regression", "Symbolic Regression",
                "Random Forest Regression", "No_Clust", "Closed path",
                "Cluster Color", "Open Path"
            ]

            st.checkbox(
                "Combine multiple regression methods?",
                key="Combine_Methods_Bt",
                help="Combine different fitting types."
            )

            # This internal logic is dynamic and does not require 'key' for saving/loading
            if st.session_state.Combine_Methods_Bt:
                selected_methods = st.multiselect(
                    "Select fitting methods to combine",
                    options=available_methods,
                    default=["Fourier"]
                )
                list_of_methods_config = []
                for method in selected_methods:
                    method_config = {"name": method}
                    if method == "Neural Network":
                        hidden_layers = st.text_input(
                            f"Hidden layers for '{method}' (e.g.: 10,10)",
                            "10,10"
                        )
                        activation = st.selectbox(
                            f"Activation for '{method}'",
                            ["relu", "tanh", "logistic"],
                            index=0
                        )
                        max_iter = st.number_input(
                            f"Max iterations for '{method}'",
                            100, 5000, step=100, value=1000
                        )
                        method_config["params"] = {
                            "hidden_layers": [int(x.strip()) for x in hidden_layers.split(",")],
                            "activation": activation,
                            "max_iter": max_iter
                        }
                    list_of_methods_config.append(method_config)
                st.sidebar.write("Selected methods:", list_of_methods_config)

            st.subheader(
                "Choose Fitting Method",
                help=(
                    "For regression, pick methods ending in -REG from the first menu; "
                    "you can combine methods and then use the slider to extend the X-axis for predictions. "
                    "For clustering, choose methods under CLUST, close paths, apply open paths, or cluster by color. "
                    "Use image editing parameters if the curve extraction isn't clear."
                )
            )

            st.selectbox(
                "fit_method_Reg",
                [
                    "No_Reg", "Linear", "Polynomial degree 2", "Moving means", "Spline",
                    "Exponential", "Fourier", "Neural Network", "Gradient Boosting",
                    "Support Vector Regression", "Symbolic Regression",
                    "Random Forest Regression"
                ],
                key="fit_method_Reg"
            )

            st.selectbox(
                "fit_method_Clust",
                ["No_Clust", "Closed path", "Cluster Color", "Open Path"],
                key="fit_method_Clust"
            )

            if st.session_state.fit_method_Clust in ["Cluster Color", "Closed path", "Open Path"]:
                st.header(
                    "⚙️ Cluster Parameter Editing",
                    help="If switching between Cluster Color and Closed path, try inverting the threshold value."
                )
                st.slider(
                    "Number of color clusters (K-Means)",
                    min_value=2, max_value=50, step=1,
                    key="N_color_clusters",
                    help="Defines how many distinct color groups to find along the curves."
                )
                st.subheader("DBSCAN")
                st.slider("DBSCAN min_samples", min_value=1, max_value=100, step=1, key="dbscan_min_samples")
                st.slider("DBSCAN eps", min_value=1, max_value=100, step=1, key="dbscan_eps")
                st.subheader("Open Paths")
                st.slider("Pixel proximity threshold for open contour", 0, 50, key="pixel_proximity_threshold")
                st.subheader("Closed Paths")
                st.slider(
                    "Offset radius for closed path perimeter",
                    min_value=1, max_value=20, key="perimeter_offset_radius",
                    help="Controls how much the perimeter deviates and its roundness."
                )
                st.slider(
                    "Smoothing sigma for closed path perimeter",
                    min_value=0.1, max_value=5.0, step=0.1,
                    key="perimeter_smoothing_sigma",
                    help="Controls the smoothness of the perimeter."
                )
                st.selectbox(
                    "Open Path Fitting Type",
                    ["Spline", "Random Forest"],
                    key="path_fit_type",
                    help="Choose the algorithm to trace the path after DBSCAN."
                )

            if st.session_state.fit_method_Reg == "Moving means":
                st.subheader("Moving Means Parameters")
                st.slider(
                    "Window size for moving means",
                    1, 50, key="window_size",
                    help="Number of points to average."
                )

            if st.session_state.fit_method_Reg == "Neural Network":
                st.subheader("Neural Network Parameters")
                st.text_input(
                    "Neurons per hidden layer (comma-separated)",
                    key="hidden_layers"
                )
                st.selectbox(
                    "Activation function",
                    ['identity', 'logistic', 'tanh', 'relu'],
                    index=2,
                    key="activation"
                )
                st.number_input(
                    "Max iterations",
                    min_value=100, max_value=100000,
                    step=100,
                    key="max_iter"
                )

            st.header("Forecasting")
            st.slider(
                "Forecast length (X-axis extension)",
                1.0, 300.0, step=0.1, key="forecast_length"
            )

            st.header("Fourier Approximation (for implicit curves)")
            st.checkbox(
                "Approximate with Fourier (if fit is implicit)",
                key="approx_fourier"
            )
            if st.session_state.approx_fourier:
                st.slider(
                    "Fourier harmonics for approximation",
                    1, 25, key="fourier_approx_harmonics"
                )

            sidebar_keys = [
            'enable_data_filter',
            'filter_window_size',
            'iqr_multiplier',
            'threshold_val',
            'invert_threshold',
            'alpha_contrast',
            'beta_brightness',
            'canny_low',
            'canny_high',
            'hough_thresh',
            'hough_min_length',
            'hough_max_gap',
            'tol_dx',
            'tol_dy',
            'center_plot',
            'x0_pix', 'x1_pix', 'x0_val', 'x1_val',
            'y0_pix', 'y1_pix', 'y0_val', 'y1_val',
            'fit_method_Reg', 'fit_method_Clust', 'N_color_clusters',
            'window_size', 'hidden_layers', 'activation', 'max_iter', 'Combine_Methods_Bt',
            'forecast_length', 'approx_fourier', 'dbscan_min_samples', 'dbscan_eps',
            'pixel_proximity_threshold', 'perimeter_offset_radius', 'perimeter_smoothing_sigma',
            'path_fit_type'
        ]        
        st.session_state.sidebar_params = {k: st.session_state[k] for k in sidebar_keys if k in st.session_state}
        #st.rerun()
    with tab1:
        st.markdown(tab_labels[0], unsafe_allow_html=True)
        with st.expander("ℹ️Info"):
            st.info(
                "Module A:" \
                "\n\n Upload an image to begin regression or clustering analysis on it:" \
                "\n - Use **Image Editing** to precisely define what you need by removing unneeded parts with the Editing commands in the *SIDEBAR*" \
                "\n - Perform **regression or clustering** with various methods, from the simplest to the most advanced; **you can combine multiple fitting methods** using residuals to achieve optimal results." \
                "\n - For each fitted curve, **extract the equation or the Fourier approximation**," \
                "\n - Gather the results by **downloading the dataframe with the parameters used, cluster characteristics, and their regressions**"
            )

    
        def create_default_image():
            # Crea una canvas bianca 300x300
            img = np.ones((300, 300, 3), dtype=np.uint8) * 255
            # Aggiungi testo rosso
            cv2.putText(img, "Default", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
            # Codifica in JPEG e restituisci bytes
            is_success, buffer = cv2.imencode(".jpg", img)
            img_bytes = buffer.tobytes()
            return img, img_bytes

        with st.expander("📁 Select input file"):
            uploaded_file = st.file_uploader(
                "Select the image file to analyze (png, jpg, jpeg)",
                type=["png", "jpg", "jpeg"]
            )

        if uploaded_file is not None:
            uploaded_file.seek(0)  # Return to the beginning of the file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = uploaded_file.name
            if img_bgr is None:
                st.error("Error decoding the uploaded image. Please try another file.")
        else:
            # Create default image
            img_bgr, img_bytes = create_default_image()
            uploaded_file = BytesIO(img_bytes)  # fake file
            uploaded_file.name = "default.jpg"
            image_name = uploaded_file.name
            st.info("No image uploaded: using default image.")

        # Now you can use img_bgr and image_name for the rest of the processing...
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=f"Image: {image_name}")





        def recenter_mask(mask):
            coords = cv2.findNonZero(mask)
            if coords is None:
                return mask
            x, y, w, h = cv2.boundingRect(coords)
            cx_mask = x + w // 2
            cy_mask = y + h // 2

            height, width = mask.shape
            cx_img = width // 2
            cy_img = height // 2

            dx = cx_img - cx_mask
            dy = cy_img - cy_mask

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_mask = cv2.warpAffine(mask, M, (width, height))
            return shifted_mask
        
        def process_image(img, canny_low, canny_high, hough_thresh, hough_min_length, hough_max_gap, tol_dx, tol_dy, threshold_val, invert_threshold, alpha_contrast=1.0, beta_brightness=0):
            global mask_curve_centered # Se questa variabile globale è ancora in uso, mantienila
            
            # --- APPLICAZIONE CONTRASTO E LUMINOSITÀ ---
            # `img` è l'immagine in formato BGR (NumPy array).
            # `alpha_contrast` (double) per il contrasto (1.0 = nessun cambiamento)
            # `beta_brightness` (int) per la luminosità (0 = nessun cambiamento)
            # cv2.convertScaleAbs esegue l'operazione: pixel_out = |alpha * pixel_in + beta|
            # e poi converte in uint8 (0-255).
            adjusted_img = cv2.convertScaleAbs(img, alpha=alpha_contrast, beta=beta_brightness)

            # Ora tutte le operazioni successive useranno `adjusted_img`
            gray = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)

            import platform
            if platform.system() == 'Windows':
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            else:
                pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


            # --- INIZIO MODIFICHE PER OCR ---
            # 1. Preparazione dell'immagine per l'OCR
            # Upscaling: ingrandiamo l'immagine per rendere i caratteri più grandi e facili da leggere
            scale_factor_ocr = 3 # Ingrandisci di 3 volte (puoi sperimentare con 2, 3 o 4)
            ocr_image_prepared = cv2.resize(gray, None, fx=scale_factor_ocr, fy=scale_factor_ocr, interpolation=cv2.INTER_CUBIC)

            # Binarizzazione: convertiamo in bianco e nero per migliorare il contrasto testo/sfondo per Tesseract
            # Usiamo THRESH_OTSU per trovare automaticamente una buona soglia
            # Il testo numerico nei grafici è spesso scuro su sfondo chiaro, quindi THRESH_BINARY_INV è utile
            _, ocr_image_prepared = cv2.threshold(ocr_image_prepared, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Aggiungi un bordo bianco per evitare che il testo ai bordi venga tagliato
            # (utile se le etichette degli assi sono molto vicine al bordo dell'immagine)
            border_size = 10
            ocr_image_prepared = cv2.copyMakeBorder(ocr_image_prepared, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)

            # 2. Configurazione di Tesseract
            ocr_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789.-'

            ocr_result = pytesseract.image_to_string(ocr_image_prepared, config=ocr_config)
            # --- FINE MODIFICHE PER OCR ---

            edges = cv2.Canny(gray, canny_low, canny_high)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                                    minLineLength=hough_min_length, maxLineGap=hough_max_gap)

            mask_lines = np.zeros_like(gray)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = x2 - x1
                    dy = y2 - y1
                    if abs(dx) < tol_dx: # Controlla se la linea è verticale
                        cv2.line(mask_lines, (x1, y1), (x2, y2), 255, 3)
                    elif abs(dy) < tol_dy: # Controlla se la linea è orizzontale
                        cv2.line(mask_lines, (x1, y1), (x2, y2), 255, 3)

            mask_lines_centered = recenter_mask(mask_lines) # Assicurati che recenter_mask sia definita e accessibile

            thresh_type = cv2.THRESH_BINARY_INV if invert_threshold else cv2.THRESH_BINARY
            _, mask_curve = cv2.threshold(gray, threshold_val, 255, thresh_type)

            mask_curve_only = cv2.bitwise_and(mask_curve, cv2.bitwise_not(mask_lines_centered))

            mask_curve_centered = recenter_mask(mask_curve_only)

            return ocr_result, mask_lines_centered, mask_curve_centered, adjusted_img # Ritorna adjusted_img per visualizzazione o ulteriori usi

        def extract_and_convert_points(mask_curve_only, x0_pix_in, x1_pix_in, x0_val_in, x1_val_in, y0_pix_in, y1_pix_in, y0_val_in, y1_val_in, enable_data_filter, filter_window_size, iqr_multiplier):
            points = cv2.findNonZero(mask_curve_only)
            if points is None:
                return None, None, None
            points = points[:, 0, :]

            # Sort points by X-coordinate to ensure proper curve progression
            sorted_indices = np.argsort(points[:, 0])
            points_sorted = points[sorted_indices]

            # Handle duplicate X values by taking the mean of Y values for common X
            df_pixels = pd.DataFrame(points_sorted, columns=['X_pix', 'Y_pix'])
            df_pixels_unique_x = df_pixels.groupby('X_pix')['Y_pix'].mean().reset_index()

            x_pix = df_pixels_unique_x['X_pix'].values
            y_pix = df_pixels_unique_x['Y_pix'].values

            # Calculate scaling factors
            scale_x = (x1_val_in - x0_val_in) / (x1_pix_in - x0_pix_in) if (x1_pix_in - x0_pix_in) != 0 else 1
            # Invert Y axis mapping because image Y increases downward
            scale_y = -(y1_val_in - y0_val_in) / (y1_pix_in - y0_pix_in) if (y1_pix_in - y0_pix_in) != 0 else 1

            # Convert pixel coordinates to real-world coordinates
            x_real = x0_val_in + (x_pix - x0_pix_in) * scale_x
            y_real = y0_val_in + (y_pix - y0_pix_in) * scale_y

            # --- INIZIO: Applicazione del Data Filter ---
            if enable_data_filter and filter_window_size > 1:
                if len(y_real) < filter_window_size:
                    st.warning(f"Insufficient number of points ({len(y_real)}) for window ({filter_window_size}). Filter disabled.")
                else:
                    # 1. Smoothing con filtro mediano
                    y_smoothed = medfilt(y_real, kernel_size=filter_window_size)

                    # 2. Rimozione Outlier basata su IQR
                    residuals = np.abs(y_real - y_smoothed)
                    Q1 = np.percentile(residuals, 25)
                    Q3 = np.percentile(residuals, 75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + iqr_multiplier * IQR
                    non_outlier_indices = residuals <= upper_bound

                    x_real = x_real[non_outlier_indices]
                    y_real = y_real[non_outlier_indices]

                    if len(x_real) == 0:
                        st.warning("All points have been removed by the filter. Try reducing the IQR multiplier or disabling the filter.")
                        return None, None, None
            # --- FINE: Applicazione del Data Filter ---

            df = pd.DataFrame({'X': x_real, 'Y': y_real})    
            return x_pix, y_pix, df


        def cosine_similarity_between_curves(fitted_x, fitted_y, mask_shape, original_mask_centered, x0_pix_in, x1_pix_in, x0_val_in, x1_val_in, y0_pix_in, y1_pix_in, y0_val_in, y1_val_in):
            # Assicurati che fitted_x e fitted_y siano liste di array, anche se contengono un solo array
            if not isinstance(fitted_x, list):
                fitted_x = [fitted_x]
            if not isinstance(fitted_y, list):
                fitted_y = [fitted_y]

            # Inizializza una lista per le similarità se ci sono più curve
            similarities = []

            for curve_x, curve_y in zip(fitted_x, fitted_y):
                if curve_x is None or curve_y is None or curve_x.size == 0 or curve_y.size == 0:
                    continue # Salta curve vuote o None

                # Crea una maschera binaria per la curva fittata di questa iterazione
                fitted_mask = np.zeros(mask_shape, dtype=np.uint8)

                # Ora itera sui singoli punti all'interno di ciascun array di curva
                for i in range(len(curve_x)):
                    x_real = curve_x[i] # x_real è ora un singolo scalare
                    y_real = curve_y[i] # y_real è ora un singolo scalare

                    # Conversione da coordinate reali a pixel
                    # Assicurati che scale_x_rev e scale_y_rev siano calcolati correttamente
                    # Li calcolo qui per chiarezza, se sono già globali o passati, riutilizzali.
                    scale_x_rev = (x1_pix_in - x0_pix_in) / (x1_val_in - x0_val_in)
                    scale_y_rev = (y1_pix_in - y0_pix_in) / (y1_val_in - y0_val_in)

                    x_img = int(x0_pix_in + (x_real - x0_val_in) * scale_x_rev)
                    y_img = int(y0_pix_in + (y_real - y0_val_in) * scale_y_rev)

                    # Assicurati che i pixel siano all'interno dei limiti dell'immagine
                    x_img = np.clip(x_img, 0, mask_shape[1] - 1)
                    y_img = np.clip(y_img, 0, mask_shape[0] - 1)

                    fitted_mask[y_img, x_img] = 255 # Disegna il pixel sulla maschera fittata

                # A questo punto, fitted_mask contiene la rappresentazione pixel della singola curva fittata
                # original_mask_centered dovrebbe essere una singola maschera, non una lista
                # Appiattisci le maschere per il calcolo della similarità coseno
                # Assicurati che original_mask_centered sia binarizzata (0 o 1) o convertila.
                # Se original_mask_centered è 0/255, normalizza:
                original_flat = (original_mask_centered / 255.0).flatten()
                fitted_flat = (fitted_mask / 255.0).flatten()

                # Calcola la similarità coseno per questa curva
                if np.linalg.norm(original_flat) == 0 or np.linalg.norm(fitted_flat) == 0:
                    # Evita divisione per zero se una delle maschere è completamente vuota
                    curve_similarity = 0.0
                else:
                    curve_similarity = np.dot(original_flat, fitted_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(fitted_flat))
                
                similarities.append(curve_similarity)

            # Se ci sono più curve fittate, puoi decidere come aggregare le similarità
            # Ad esempio, restituire la media o il massimo, o una lista
            if similarities:
                return np.mean(similarities) # Restituisce la media delle similarità
            else:
                return 0.0 # Nessuna similarità calcolata

        def write_equation(method, params=None, approx_type=None):
            """
            Generates and displays the equation for the fitted curve.
            Adds a parameter `approx_type` to differentiate the main fit from a Fourier approximation.

            Args:
                method (str): The chosen fitting method.
                params (tuple/object): Parameters derived from the curve fitting.
                approx_type (str, optional): Type of approximation if applicable (e.g., "Fourier Approximation"). Defaults to None.
            """
            if approx_type:
                st.subheader(f"📝 {approx_type} of the fitted curve")
            else:
                st.subheader("📝 Equation of the fitted curve")

            x_sym = sympy.symbols('x')

            if method == "Linear":
                if params is not None and len(params) == 2:
                    m, c = params
                    st.latex(f"y = {m:.4f}{x_sym} + {c:.4f}")
                else:
                    st.info("Linear Equation: $y = mx + c$ (coefficients not available)")

            elif method == "Polinomial grade 2":
                if params is not None and len(params) == 3:
                    a, b, c = params
                    st.latex(f"y = {a:.4f}{x_sym}^2 + {b:.4f}{x_sym} + {c:.4f}")
                else:
                    st.info("Polinomial grade 2 Equation: $y = ax^2 + bx + c$ (coefficients not available)")

            elif method == "Exponential":
                if params is not None and len(params) == 3:
                    a, b, c = params
                    st.latex(f"y = {a:.4f}e^{{{b:.4f}{x_sym}}} + {c:.4f}")
                else:
                    st.info("Exponential Equation: $y = ae^{bx} + c$ (coefficients not available)")


            elif method == "Fourier":
                if params is not None and len(params) > 1:
                    # Fourier params here include L (period) as the first element if optimized
                    if approx_type: # For optimized Fourier approximation, L is part of params
                        L = params[0]
                        coeffs = params[1:] # Remaining are a0, a1, b1, a2, b2, ...
                        N = (len(coeffs) - 1) // 2
                        equation_str = f"y = {coeffs[0]/2:.4f}"
                        for n in range(1, N + 1):
                            equation_str += f" + {coeffs[2*n-1]:.4f}\\cos({n} \\frac{{2\\pi}}{{{L:.4f}}}{x_sym}) + {coeffs[2*n]:.4f}\\sin({n} \\frac{{2\\pi}}{{{L:.4f}}}{x_sym})"
                        st.latex(equation_str)
                    else: # Regular Fourier fit
                        N = (len(params) - 1) // 2
                        equation_str = f"y = {params[0]/2:.4f}"
                        # For regular Fourier, we assume the input x was already scaled to [0, 2pi] relative to its range.
                        # Here, we need to know the 'L' used for scaling in fit_curve to reconstruct the equation properly.
                        # For simplicity, if params only contains coefficients, assume it's for x in [0, 2pi].
                        # A more robust solution would pass L as well.
                        # For now, let's just indicate a generic x without scaling if L isn't explicitly passed here for the original Fourier fit.
                        st.warning("The Fourier equation for direct fitting requires additional information about the period to be displayed correctly here.")
                        st.latex("y = A_0/2 + \\sum (A_n\\cos(nx) + B_n\\sin(nx))")
                else:
                    st.info("Fourier equation: $y = a_0/2 + \\sum_{n=1}^{N} (a_n\\cos(nx) + b_n\\sin(nx))$ (coefficienti non disponibili)")

            elif method == "Symbolic Regression":
                if params is not None and hasattr(params, 'sympy_formula'):
                    st.latex(f"y = {sympy.latex(params.sympy_formula())}")
                else:
                    st.info(
                        "The symbolic regression equation will be displayed directly in the fitting output "
                        "(if PySR is enabled and finds a formula)."
                    )

            elif method in [
                "Spline", "Neural Network", "Random Forest Regression",
                "Gradient Boosting", "Support Vector Regression", "Closed path"
            ]:
                if not approx_type:  # Only show this message if it's the main fit, not an approximation
                    st.info(
                        f"The method '{method}' does not produce an explicit equation that can be easily displayed. "
                        "It's a model based on complex algorithms."
                    )
            else:
                st.info(
                    "No fitting method selected or no equation available for the chosen method."
                )

        # --- New Fourier Approximation Function ---
        def fourier_series_with_period(x_vals, L, *a_coeffs):
            """Fourier series function where L is the period and the first coefficient is a0/2."""
            ret = a_coeffs[0] / 2
            for n in range(1, (len(a_coeffs) - 1) // 2 + 1):
                ret += (
                    a_coeffs[2 * n - 1] * np.cos(n * 2 * np.pi * x_vals / L)
                    + a_coeffs[2 * n] * np.sin(n * 2 * np.pi * x_vals / L)
                )
            return ret

        def approximate_curve_with_fourier(x_data, y_data, n_harmonics=5):
            """
            Approximates given x_data, y_data with a Fourier series,
            optimizing for coefficients and period (L).
            Returns x_fit, y_approx_fourier, and the optimized parameters including L.
            """
            if len(x_data) < 2 * n_harmonics + 2:  # need at least this many points for N harmonics + L
                st.warning(
                    f"Too few points to approximate with {n_harmonics} Fourier harmonics. "
                    f"At least {2 * n_harmonics + 2} points are required."
                )
                return None, None, None

            # Initial guess for L (period) can be the range of the data
            initial_L_guess = x_data.max() - x_data.min()
            if initial_L_guess <= 0:
                initial_L_guess = 1.0 # Fallback for constant or single point data

            # Initial guess for Fourier coefficients (a0, a1, b1, ...)
            # Number of coefficients: 1 (a0) + 2*N_harmonics (a_n, b_n)
            initial_coeffs_guess = [0.0] * (1 + 2 * n_harmonics)

            # Combine L and coefficients into one parameter array for curve_fit
            p0 = [initial_L_guess] + initial_coeffs_guess

            try:
                # Define bounds for parameters
                # L must be > 0
                # Coefficients can be anything for now, but can be bounded if needed
                bounds_lower = [1e-6] + [-np.inf] * (1 + 2 * n_harmonics)
                bounds_upper = [np.inf] + [np.inf] * (1 + 2 * n_harmonics)

                popt, pcov = curve_fit(fourier_series_with_period, x_data, y_data, p0=p0,
                                    bounds=(bounds_lower, bounds_upper), maxfev=20000)

                L_opt = popt[0]
                coeffs_opt = popt[1:]

                x_approx_fourier = np.linspace(x_data.min(), x_data.max()+ st.session_state.forecast_length, 500)
                y_approx_fourier = fourier_series_with_period(x_approx_fourier, L_opt, *coeffs_opt)
                st.session_state.y_approx_fourier = y_approx_fourier
                return x_approx_fourier, y_approx_fourier, popt # Return all optimized params
            except RuntimeError as e:
                st.warning(f"Unable to find Fourier approximation (optimization failed): {e}. Try reducing the number of harmonics or choosing another method.")
                return None, None, None
            except Exception as e:
                st.warning(f"Error during Fourier approximation: {e}")
                return None, None, None

        
        def fit_curve(df, method, forecast_length, hidden_layers_val, activation_val, max_iter_val, mask_curve_only=None, original_image_rgb=None):
            """
            Fitta una curva ai dati forniti usando il metodo specificato e restituisce
            i dati fittati e i parametri dell'equazione.
            """
            x = df["X"].values
            y = df["Y"].values

            sorted_indices = np.argsort(x)
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]

            _, unique_indices = np.unique(x_sorted, return_index=True)
            x_unique = x_sorted[unique_indices]
            y_unique = y_sorted[unique_indices]
            x_max = x_unique.max()
            x_min = x_unique.min()

            x_extended = np.linspace(x_min, x_max + forecast_length, 500)
            st.session_state.x_extended =x_extended
            params = None

            if method == "Spline":
                s_val = 0.5 * np.var(y_unique) * len(y_unique)
                spline = UnivariateSpline(x_unique, y_unique, s=s_val)
                y_fit = spline(x_extended)
                return x_extended, y_fit, params

            elif method == "Polinomial grade 2":
                p = np.polyfit(x_unique, y_unique, 2)
                y_fit = np.polyval(p, x_extended)
                params = p
                return x_extended, y_fit, params

            elif method == "Linear":
                p = np.polyfit(x_unique, y_unique, 1)
                y_fit = np.polyval(p, x_extended)
                params = p
                return x_extended, y_fit, params
            elif method == "Movie means":
                # Assicurati che 'window_size' sia accessibile qui.
                # Se 'window_size' viene passato direttamente alla funzione che chiama fit_curve,
                # allora dovrebbe essere disponibile in questo scope.
                # Esempio: Assicurati che sia una variabile globale o passata in qualche modo.
                # PER QUESTO ESEMPIO, STO ASSUMENDO CHE 'window_size' SIA GIA' DEFINITO E ACCESSIBILE
                # COME UNA VARIABILE NEL TUO SCRIPT PRINCIPALE CHE CHIAMERÀ QUESTA FUNZIONE.
                # Se non lo è, dovrai passarlo come parametro a 'fit_curve'.
                # Per esempio:
                # if 'window_size' not in globals(): # Questo è un check rudimentale
                #     st.error("Errore: La dimensione della finestra (window_size) non è stata definita.")
                #     return x_data, y_data, np.array([]) # Ritorno di default in caso di errore

                # Validazione della finestra, se non già fatta nello slider
                if not isinstance(st.session_state.window_size, (int, float)) or st.session_state.window_size < 1:
                    st.error("Error: Invalid moving means window size. Please make sure the slider is set.")
                    params = np.array([]) # Nessun parametro valido
                    return x_extended, y_fit, params # Ritorna dati originali in caso di errore

                # Calcola la Movie means sui dati y_unique per il fit
                # Si assume che x_unique e y_unique siano le coppie di punti uniche per il fitting
                fitted_series = pd.Series(y_unique).rolling(window=int(st.session_state.window_size), min_periods=1, center=False).mean()
                fitted_curve_y = fitted_series.values # La curva fittata è la serie della Movie means sui punti unici

                # La curva estesa (e per la previsione)
                # Per la Movie means, l'estensione è spesso la media dell'ultimo blocco,
                # o un'estrapolazione semplice. Qui useremo l'ultimo valore valido calcolato.
                if len(fitted_curve_y) > 0:
                    last_ma_value = fitted_curve_y[-1]
                else:
                    last_ma_value = np.mean(y_unique) if len(y_unique) > 0 else 0

                # Crea la y_fit per l'estensione completa (inclusa la previsione)
                # Per la Movie means, il fit sull'estensione è solitamente l'ultimo valore MA
                y_fit = np.full(x_extended.shape, last_ma_value)
                # Sostituisci i valori già calcolati con la Movie means effettiva
                if len(x_unique) > 0:
                    for i, x_val in enumerate(x_unique):
                        idx_extended = np.where(x_extended == x_val)[0]
                        if len(idx_extended) > 0:
                            y_fit[idx_extended[0]] = fitted_curve_y[i]

                # Per i parametri della Movie means, possiamo restituire la dimensione della finestra
                # in un array NumPy per coerenza.
                params = np.array([float(st.session_state.window_size)]) # Contiene la dimensione della finestra
                return x_extended, y_fit, params

            elif method == "Exponential":
                def exp_func(x, a, b, c):
                    return a * np.exp(b * x) + c
                try:
                    if len(x_unique) >= 2:
                        # Initial guess for exponential based on properties
                        a_guess = (y_unique[-1] - y_unique[0]) / (np.exp(x_unique[-1]) - np.exp(x_unique[0])) if (np.exp(x_unique[-1]) - np.exp(x_unique[0])) != 0 else 1.0
                        b_guess = np.log((y_unique[-1] - np.min(y_unique) + 1e-9) / (y_unique[0] - np.min(y_unique) + 1e-9)) / (x_unique[-1] - x_unique[0] + 1e-9) if (x_unique[-1] - x_unique[0]) != 0 else 0.1
                        c_guess = np.min(y_unique) - 1e-9 # Slightly below min value
                        p0_guess = [a_guess, b_guess, c_guess]

                        # Ensure initial guess is finite
                        p0_guess = [np.nan_to_num(v, nan=1.0, posinf=1.0, neginf=-1.0) for v in p0_guess]
                    else:
                        p0_guess = [1, 0.1, 0] # Default if not enough points

                    popt, _ = curve_fit(exp_func, x_unique, y_unique, p0=p0_guess, maxfev=10000)
                    y_fit = exp_func(x_extended, *popt)
                    params = popt
                    return x_extended, y_fit, params
                except Exception as e:
                    st.warning(f" Exponential fit failed: {e}. Try another method or check your data.")
                    return x_unique, y_unique, None

            elif method == "Fourier":
                N = 5
                L = x_unique.max() - x_unique.min()
                if L == 0:
                    st.warning("X range for Fourier is zero. Cannot fit.")
                    return x_unique, y_unique, None

                x_scaled = 2 * np.pi * (x_unique - x_unique.min()) / L

                def fourier_series(x_vals, *a_coeffs):
                    ret = a_coeffs[0] / 2
                    for n in range(1, N + 1):
                        ret += a_coeffs[2 * n - 1] * np.cos(n * x_vals) + a_coeffs[2 * n] * np.sin(n * x_vals)
                    return ret

                initial_guess = np.zeros(2 * N + 1)
                try:
                    popt, _ = curve_fit(fourier_series, x_scaled, y_unique, p0=initial_guess)
                    x_fit_scaled = 2 * np.pi * (x_extended - x_unique.min()) / L
                    y_fit = fourier_series(x_fit_scaled, *popt)
                    params = popt # Here, params are just coefficients, not including L explicitly for the formula
                    return x_extended, y_fit, params
                except Exception as e:
                    st.warning(f"Fourier Fit failed: {e}. Try another method or check your data.")
                    return x_unique, y_unique, None

            elif method == "Neural Network":
                try:
                    hidden_layer_sizes_tuple = tuple(int(n.strip()) for n in hidden_layers_val.split(",") if n.strip())
                    if len(hidden_layer_sizes_tuple) == 0:
                        hidden_layer_sizes_tuple = (8,)
                except Exception:
                    hidden_layer_sizes_tuple = (8,)

                mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes_tuple,
                                activation=activation_val,
                                max_iter=max_iter_val,
                                random_state=1)
                x_reshape = x_unique.reshape(-1, 1)
                try:
                    mlp.fit(x_reshape, y_unique)
                    x_extended_reshaped = x_extended.reshape(-1, 1)
                    y_fit = mlp.predict(x_extended_reshaped)
                    return x_extended_reshaped.flatten(), y_fit, None
                except Exception as e:
                    st.warning(f"Neural Network fit failed: {e}")
                    return x_unique, y_unique, None

            elif method == "Closed path":
                if mask_curve_only is None or mask_curve_only.size == 0:
                    st.warning("Binarized curve image missing or empty for Closed path.")
                    return np.array([]), np.array([]), None

                if np.count_nonzero(mask_curve_only) == 0:
                    st.warning("No white pixels detected. The image contains no visible contours.")
                    return np.array([]), np.array([]), None

                # Find all contours
                contours, _ = cv2.findContours(mask_curve_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if not contours:
                    st.warning("No contour detected.")
                    return np.array([]), np.array([]), None

                # Combine all contour points into a single array for clustering
                all_contour_points = []
                for contour in contours:
                    # Ensure the contour has at least one point
                    if contour.shape[0] >= 1:
                        # Squeeze the array, then convert to list.
                        # If squeeze reduces to 1D (a single point), ensure it's a list of lists.
                        squeezed_points = contour.squeeze()
                        if squeezed_points.ndim == 1:  # Single point [x, y]
                            all_contour_points.append(squeezed_points.tolist())
                        else:  # Multiple points [[x1,y1], [x2,y2], ...]
                            all_contour_points.extend(squeezed_points.tolist())

                if not all_contour_points:
                    st.warning("No valid points found in contours for clustering.")
                    return np.array([]), np.array([]), None

                # Now points_for_clustering will be a consistent 2D array (N, 2)
                points_for_clustering = np.array(all_contour_points)

                if len(points_for_clustering) < 2:
                    st.warning("Insufficient points to perform DBSCAN clustering.")
                    return np.array([]), np.array([]), None


                # Esegui DBSCAN sui punti dei contorni
                db = DBSCAN(eps=st.session_state.dbscan_eps, min_samples=st.session_state.dbscan_min_samples).fit(points_for_clustering)
                labels = db.labels_

                # Crea un'immagine RGB per visualizzare i contorni clusterizzati colorati
                #clustered_image = original_image_rgb.copy()
                clustered_image = cv2.cvtColor(mask_curve_only * 255, cv2.COLOR_GRAY2BGR)
            
                # Inizializza liste per i dati fittati di tutti i cluster
                all_fitted_x_clusters = []
                all_fitted_y_clusters = []
                cluster_info_list = [] # Per salvare informazioni su ogni cluster

                # Mappa le etichette dei cluster ai colori casuali
                unique_labels = set(labels)
                st.session_state.cluster_labels = np.array(list(unique_labels))
                
                colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
                # Rimuovi il nero per il rumore se presente
                if -1 in unique_labels:
                    colors.pop(list(unique_labels).index(-1)) # Rimuovi il colore assegnato al rumore

                # Processa ogni cluster
                for k, col in zip(unique_labels, colors):
                    if k == -1: # Rumore, lo ignoriamo o lo disegniamo in un colore specifico (es. nero/grigio)
                        cluster_color = (50, 50, 50) # Grigio scuro per il rumore
                        cluster_label = "Noise"
                    else:
                        cluster_points_indices = (labels == k)
                        cluster_points_pix = points_for_clustering[cluster_points_indices]

                        if len(cluster_points_pix) < 2: # Assicurati che ci siano abbastanza punti per fittare
                            continue

                        # Disegna i punti del cluster sull'immagine colorata
                        # Converti il colore da [0,1] a [0,255] e da RGBA a BGR
                        cluster_color = (int(col[2]*255), int(col[1]*255), int(col[0]*255))
                        for pt in cluster_points_pix:
                            cv2.circle(clustered_image, tuple(pt), 1, cluster_color, -1) # Disegna i pixel del cluster

                        # Converti i punti del cluster da pixel a valori reali
                        x_cluster_pix = cluster_points_pix[:, 0]
                        y_cluster_pix = cluster_points_pix[:, 1]

                        # Applica le stesse trasformazioni di scala
                        # Utilizza st.session_state per accedere ai parametri di calibrazione degli assi
                        # Assicurati che questi valori siano impostati in st.session_state prima di chiamare fit_curve
                        scale_x = (st.session_state.x1_val - st.session_state.x0_val) / (st.session_state.x1_pix - st.session_state.x0_pix + 1e-9)
                        scale_y = (st.session_state.y1_val - st.session_state.y0_val) / (st.session_state.y1_pix - st.session_state.y0_pix + 1e-9)

                        x_cluster_real = st.session_state.x0_val + (x_cluster_pix - st.session_state.x0_pix) * scale_x
                        y_cluster_real = st.session_state.y0_val + (y_cluster_pix - st.session_state.y0_pix) * scale_y

                        # Ordina e gestisci i duplicati per il fitting
                        df_cluster = pd.DataFrame({'X': x_cluster_real, 'Y': y_cluster_real})
                        df_cluster_unique_x = df_cluster.groupby('X')['Y'].mean().reset_index()
                        x_cluster_unique = df_cluster_unique_x['X'].values
                        y_cluster_unique = df_cluster_unique_x['Y'].values

                        if len(x_cluster_unique) < 2:
                            st.warning(f"Cluster {k}: insufficient points for fitting after duplicate removal.")
                            continue
                        else:
                            try:
                                s_val = 0.5 * np.var(y_cluster_unique) * len(y_cluster_unique)
                                if s_val == 0:
                                    s_val = 1e-6  # Avoid division by zero if all Y are equal
                                spline = UnivariateSpline(x_cluster_unique, y_cluster_unique, s=s_val)
                                x_fit_cluster = np.linspace(
                                    x_cluster_unique.min(),
                                    x_cluster_unique.max() + forecast_length,
                                    200
                                )
                                y_fit_cluster = spline(x_fit_cluster)

                                all_fitted_x_clusters.append(x_fit_cluster)
                                all_fitted_y_clusters.append(y_fit_cluster)
                                cluster_info_list.append({
                                    "label": f"Cluster {k}",
                                    "color": col,
                                    "equation": "Spline (non-explicit)"
                                })

                            except Exception as e:
                                st.warning(f"Spline fitting failed for Cluster {k}: {e}")
                                continue

                # Display the image with colored clusters
                st.image(
                    clustered_image,
                    caption="Clustered Contours",
                    channels="BGR",
                    use_container_width=True
                )

                # Display cluster details
                st.subheader("📊 Cluster Details")
                if cluster_info_list:
                    for info in cluster_info_list:
                        st.write(f"- **{info['label']}** (color: RGB{tuple(int(c*255) for c in info['color'][:3])})")
                        st.write(f"  Equation: {info['equation']}")

                    # Save metadata for future use (e.g., exporting a DataFrame)
                    st.session_state.cluster_metadata = [
                        {
                            "label": info["label"],
                            "color_rgb": tuple(int(c*255) for c in info["color"][:3]),
                            "equation": info["equation"]
                        }
                        for info in cluster_info_list
                    ]
                    # Compatibility for downstream use
                    st.session_state.colore_cluster = [
                        meta["color_rgb"] for meta in st.session_state.cluster_metadata
                    ]
                else:
                    st.info("No significant clusters detected for fitting.")

                # Return all fitted cluster data
                return all_fitted_x_clusters, all_fitted_y_clusters, None

            elif method == "Cluster Color":
                if mask_curve_only.shape != original_image_rgb.shape[:2]:
                    mask_curve_only = cv2.resize(
                        mask_curve_only,
                        (original_image_rgb.shape[1], original_image_rgb.shape[0])
                    )
                # Initial mask checks
                if mask_curve_only is None or mask_curve_only.size == 0:
                    st.warning("Binarized curve image missing or empty for Cluster Color.")
                    return np.array([]), np.array([]), None

                if np.count_nonzero(mask_curve_only) == 0:
                    st.warning("No white pixels detected. The image contains no visible contours for color clustering.")
                    return np.array([]), np.array([]), None

                # Ensure original color image is available
                if original_image_rgb is None:
                    st.error("Original color image (original_image_rgb) is not available. Please upload an image.")
                    return np.array([]), np.array([]), None

                # Find all curve pixels (white in the binary mask)
                pixel_coords_raw = cv2.findNonZero(mask_curve_only)
                if pixel_coords_raw is None or len(pixel_coords_raw) == 0:
                    st.warning("No curve pixels detected. Check the binarization threshold.")
                    return [], [], None

                # Flatten pixel coordinates (shape (N, 2))
                all_curve_pixels = pixel_coords_raw[:, 0, :]

                if len(all_curve_pixels) < 2:
                    st.warning("Insufficient points to perform color-based clustering.")
                    return np.array([]), np.array([]), None

                # Filter pixels that lie within the original image borders
                valid_indices = (
                    (all_curve_pixels[:, 1] >= 0) &
                    (all_curve_pixels[:, 1] < original_image_rgb.shape[0]) &
                    (all_curve_pixels[:, 0] >= 0) &
                    (all_curve_pixels[:, 0] < original_image_rgb.shape[1])
                )
                if np.sum(valid_indices) == 0:
                    st.warning("No valid pixels found within the original image boundaries for color clustering.")
                    return np.array([]), np.array([]), None

                all_curve_pixels_valid = all_curve_pixels[valid_indices]
                # Extract RGB values for each curve pixel
                pixel_colors = original_image_rgb[
                    all_curve_pixels_valid[:, 1],
                    all_curve_pixels_valid[:, 0]
                ]

                # Number of color clusters from Streamlit state
                n_color_clusters = st.session_state.N_color_clusters

                if len(pixel_colors) < n_color_clusters:
                    st.warning(
                        f"Too few curve pixels ({len(pixel_colors)}) to create "
                        f"{n_color_clusters} color clusters. Reduce the number of clusters or check the binarization."
                    )
                    return np.array([]), np.array([]), None

                # Perform K-Means on the colors
                kmeans = KMeans(n_clusters=n_color_clusters, random_state=0, n_init=10)
                labels = kmeans.fit_predict(pixel_colors)

                # Map K-Means labels to RGB centroids
                cluster_centers_rgb = kmeans.cluster_centers_.astype(int)
                st.session_state.colore_cluster = cluster_centers_rgb
                # Prepare image for colored-cluster visualization
                clustered_image = cv2.cvtColor(st.session_state.mask_curve_only * 255, cv2.COLOR_GRAY2BGR)

                all_fitted_x_clusters = []
                all_fitted_y_clusters = []
                cluster_info_list = []

                unique_labels = np.unique(labels)
                for k in unique_labels:
                    cluster_points_indices = (labels == k)
                    cluster_points_pix = all_curve_pixels_valid[cluster_points_indices]

                    if len(cluster_points_pix) < 2:  # Need at least 2 points for fitting
                        continue

                    # Draw cluster pixels using centroid color (convert RGB to BGR)
                    color_bgr = (
                        int(cluster_centers_rgb[k][2]),
                        int(cluster_centers_rgb[k][1]),
                        int(cluster_centers_rgb[k][0])
                    )
                    for pt in cluster_points_pix:
                        cv2.circle(clustered_image, tuple(pt), 1, color_bgr, -1)

                    # Convert pixel coordinates to real-world values
                    x_pix = cluster_points_pix[:, 0]
                    y_pix = cluster_points_pix[:, 1]
                    scale_x = (st.session_state.x1_val - st.session_state.x0_val) / (
                        st.session_state.x1_pix - st.session_state.x0_pix + 1e-9
                    )
                    scale_y = (st.session_state.y1_val - st.session_state.y0_val) / (
                        st.session_state.y1_pix - st.session_state.y0_pix + 1e-9
                    )
                    st.session_state.scale_x = scale_x
                    st.session_state.scale_y = scale_y

                    x_real = st.session_state.x0_val + (x_pix - st.session_state.x0_pix) * scale_x
                    y_real = st.session_state.y0_val + (y_pix - st.session_state.y0_pix) * scale_y

                    # Handle duplicates for fitting
                    df_cluster = pd.DataFrame({'X': x_real, 'Y': y_real})
                    df_unique = df_cluster.groupby('X')['Y'].mean().reset_index()
                    x_unique = df_unique['X'].values
                    y_unique = df_unique['Y'].values

                    if len(x_unique) < 2:
                        st.warning(f"Cluster Color {k}: insufficient points for fitting after duplicate removal.")
                        continue

                    # Fit with a spline
                    try:
                        s_val = 0.5 * np.var(y_unique) * len(y_unique)
                        if s_val == 0:
                            s_val = 1e-6
                        spline = UnivariateSpline(x_unique, y_unique, s=s_val)
                        x_fit = np.linspace(x_unique.min(), x_unique.max() + forecast_length, 200)
                        y_fit = spline(x_fit)

                        all_fitted_x_clusters.append(x_fit)
                        all_fitted_y_clusters.append(y_fit)
                        cluster_info_list.append({
                            "label": f"Cluster Color {k}",
                            "color_rgb": tuple(cluster_centers_rgb[k]),
                            "equation": "Spline (non-explicit)"
                        })
                    except Exception as e:
                        st.warning(f"Spline fitting failed for Cluster Color {k}: {e}")
                        continue

                # Show the colored-cluster image and details
                st.image(
                    clustered_image,
                    caption="Color-Clustered Curves",
                    channels="BGR",
                    use_container_width=True
                )
                st.subheader("📊 Cluster Details (Color-Based)")
                if cluster_info_list:
                    for info in cluster_info_list:
                        st.write(f"- **{info['label']}** (color: RGB{info['color_rgb']})")
                        st.write(f"  Equation: {info['equation']}")
                else:
                    st.info("No significant clusters detected for color-based fitting.")
                st.session_state.cluster_metadata = cluster_info_list

                return all_fitted_x_clusters, all_fitted_y_clusters, None

            elif method == "Random Forest Regression":
                rf = RandomForestRegressor(n_estimators=300, random_state=42)
                rf.fit(x_unique.reshape(-1, 1), y_unique)
                y_fit = rf.predict(x_extended.reshape(-1, 1))
                return x_extended, y_fit, None

            elif method == "Gradient Boosting":
                gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
                gb.fit(x_unique.reshape(-1, 1), y_unique)
                y_fit = gb.predict(x_extended.reshape(-1, 1))
                return x_extended, y_fit, None

            elif method == "Support Vector Regression":
                svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
                svr.fit(x_unique.reshape(-1, 1), y_unique)
                y_fit = svr.predict(x_extended.reshape(-1, 1))
                return x_extended, y_fit, None

            elif method == "Symbolic Regression":
                try:
                    from pysr import PySRRegressor
                    pysr_available = True
                except ImportError:
                    pysr_available = False

                if not pysr_available:
                    st.warning("PySR is not installed. Install it with 'pip install pysr'.")
                    return x_unique, y_unique, None
                try:
                    model = PySRRegressor(
                        niterations=1000,
                        binary_operators=["+", "-", "*", "/"],
                        unary_operators=["cos", "sin", "exp", "log", "sqrt"],
                        verbosity=0,
                        random_state=42
                    )
                    model.fit(x_unique.reshape(-1, 1), y_unique)
                    y_fit = model.predict(x_extended.reshape(-1, 1))
                    params = model
                    st.write("Found formula:", model.sympy_formula())
                    return x_extended, y_fit, params
                except Exception as e:
                    st.warning(f"Symbolic regression failed: {e}")
                    return x_unique, y_unique, None
            else:
                return x_unique, y_unique, None


        def combine_fits_iteratively(
            df,
            list_of_methods_config,
            forecast_length,
            hidden_layers_val,
            activation_val,
            max_iter_val,
            mask_curve_only=None, 
            original_image_rgb=None,
            window_size=2,
            dbscan_eps=None,
            dbscan_min_samples=None,
            num_color_clusters_kmeans=3
        ):
            
            """
            Combina iterativamente fit successivi sui residui dei fit precedenti.
            Restituisce (x_fit_combinato, y_fit_combinato, combined_params) se almeno un fit ha successo,
            dove combined_params è una stringa descrittiva.
            NON restituisce nulla (la funzione termina) se nessun fit valido è stato combinato.
            """
            if not list_of_methods_config:
                st.warning("Nessun metodo di fit selezionato per la combinazione.")
                return 

            x_original = df["X"].values
            y_original = df["Y"].values

            if len(x_original) == 0:
                st.warning("Dati originali vuoti, impossibile combinare i fit.")
                return 

            x_min_global = x_original.min()
            x_max_global = x_original.max()
            global_x_extended = np.linspace(x_min_global, x_max_global + forecast_length, 500)

            combined_fitted_y = np.zeros_like(global_x_extended)
            current_residuals_y = y_original.copy()
            
            description_parts = []
            
            _hidden_layers_val = hidden_layers_val
            _activation_val = activation_val
            _max_iter_val = max_iter_val
            _window_size = window_size
            _dbscan_eps = dbscan_eps
            _dbscan_min_samples = dbscan_min_samples
            _num_color_clusters_kmeans = num_color_clusters_kmeans

            for i, method_config in enumerate(list_of_methods_config):
                method_name = method_config['name']
                method_specific_params = method_config.get('params', {})

                current_hidden_layers_val = method_specific_params.get('hidden_layers', _hidden_layers_val)
                current_activation_val = method_specific_params.get('activation', _activation_val)
                current_max_iter_val = method_specific_params.get('max_iter', _max_iter_val)
                current_window_size = method_specific_params.get('window_size', _window_size)
                current_dbscan_eps = method_specific_params.get('dbscan_eps', _dbscan_eps)
                current_dbscan_min_samples = method_specific_params.get('min_samples', _dbscan_min_samples) # Changed to 'min_samples'
                current_num_color_clusters_kmeans = method_specific_params.get('num_color_clusters', _num_color_clusters_kmeans) # Changed to 'num_color_clusters'


                df_for_current_fit = pd.DataFrame({
                    "X": x_original, 
                    "Y": current_residuals_y 
                })
                
                result_from_fit_curve = fit_curve(
                        df=df_for_current_fit,
                        method=method_name,
                        forecast_length=forecast_length,
                        hidden_layers_val=current_hidden_layers_val,
                        activation_val=current_activation_val,
                        max_iter_val=current_max_iter_val,
                        mask_curve_only=st.session_state.mask_curve_only,     
                        original_image_rgb=original_image_rgb,                
                        
                    )
                        
                if result_from_fit_curve is None:
                    st.warning(f"The fit for component '{method_name}' failed. Skipping this component.")
                    continue


                x_fit_component_raw, y_fit_component_raw, _ = result_from_fit_curve

                y_fit_component_aligned = np.interp(global_x_extended, x_fit_component_raw.flatten(), y_fit_component_raw, left=y_fit_component_raw[0], right=y_fit_component_raw[-1])


                # Regola il componente: sottrai la media se non è il primo componente e non è un metodo di clustering
                if i > 0 and method_name not in ["Closed path", "Cluster Color"]:
                    mean_to_remove = np.mean(y_fit_component_aligned)
                    y_fit_component_final = y_fit_component_aligned - mean_to_remove
                    description_parts.append(f"+ ({method_name} - media)")
                else:
                    y_fit_component_final = y_fit_component_aligned
                    description_parts.append(method_name)

                combined_fitted_y += y_fit_component_final
                
                y_combined_at_original_x = np.interp(x_original, global_x_extended, combined_fitted_y, left=combined_fitted_y[0], right=combined_fitted_y[-1])
                current_residuals_y = y_original - y_combined_at_original_x

            if not description_parts: 
                st.warning("No combine fit generated.")
                return 
            
            combined_description = " + ".join(description_parts)
            
            # Restituisce gli stessi tre valori di fit_curve
            return global_x_extended, combined_fitted_y, None

        def process_image_and_get_df(
            img_bgr,
            params,
            image_name="img",
            n_harmonics=None
        ):
        

            # Estrai tutti i parametri possibili dal dizionario
            def get(name, default=None, tp=None):
                val = params.get(name, default)
                if tp is not None:
                    try: val = tp(val)
                    except: val = default
                return val

            # --- PARAMETRI ---
            canny_low = get("Param_Canny_Low", 50, int)
            canny_high = get("Param_Canny_High", 150, int)
            hough_thresh = get("Param_Hough_Threshold", 100, int)
            hough_min_length = get("Param_Hough_Min_Length", 100, int)
            hough_max_gap = get("Param_Hough_Max_Gap", 20, int)
            tol_dx = get("Param_Tol_DX", 10, int)
            tol_dy = get("Param_Tol_DY", 10, int)
            threshold_val = get("Param_Threshold_Value", 200, int)
            invert_threshold = get("Param_Invert_Threshold", True, bool)
            alpha_contrast = get("Param_Alpha_Contrast", 1.0, float)
            beta_brightness = get("Param_Beta_Brightness", 0.0, float)
            x0_pix = get("Param_X0_Pix", 0, int)
            x1_pix = get("Param_X1_Pix", 100, int)
            x0_val = get("Param_X0_Val", 0.0, float)
            x1_val = get("Param_X1_Val", 10.0, float)
            y0_pix = get("Param_Y0_Pix", 0, int)
            y1_pix = get("Param_Y1_Pix", 100, int)
            y0_val = get("Param_Y0_Val", 0.0, float)
            y1_val = get("Param_Y1_Val", 10.0, float)
            enable_data_filter = get("Param_Enable_Data_Filter", False, bool)
            filter_window_size = get("Param_Filter_Window_Size", 7, int)
            iqr_multiplier = get("Param_IQR_Multiplier", 1.5, float)
            fit_method_Reg = get("Param_Fit_Method_Reg", "No_Reg", str)
            fit_method_Clust = get("Param_Fit_Method_Clust", "No_Clust", str)
            window_size = get("Param_Window_Size_MA", 5, int)
            hidden_layers = get("Param_NN_Hidden_Layers", "8", str)
            activation = get("Param_NN_Activation", "relu", str)
            max_iter = get("Param_NN_Max_Iter", 500, int)
            forecast_length = get("Param_Forecast_Length", 0.0, float)
            approx_fourier = get("Param_Approx_Fourier", False, bool)
            N_color_clusters = get("Param_N_Color_Clusters", 3, int)
            dbscan_min_samples = get("Param_DBSCAN_Min_Samples", 5, int)
            dbscan_eps = get("Param_DBSCAN_EPS", 5.0, float)
            fourier_approx_harmonics = get("Param_Fourier_Approx_Harmonics", 5, int)
            if n_harmonics is not None: fourier_approx_harmonics = n_harmonics

            # --- 2. Process image, get mask/curve ---
            ocr_result, mask_lines, mask_curve_only, adjusted_img = process_image(
                img_bgr, canny_low, canny_high, hough_thresh, hough_min_length,
                hough_max_gap, tol_dx, tol_dy, threshold_val, invert_threshold,
                alpha_contrast, beta_brightness
            )

            # --- 3. Extract points ---
            x_pix, y_pix, df_points = extract_and_convert_points(
                mask_curve_only, x0_pix, x1_pix, x0_val, x1_val, y0_pix, y1_pix,
                y0_val, y1_val, enable_data_filter, filter_window_size, iqr_multiplier
            )
            if df_points is None or df_points.empty:
                return pd.DataFrame({"image": [image_name], "error": ["nessun punto rilevato"]})

            # --- 4. Fitting primario (reg) ---
            x_fit, y_fit, fit_params = fit_curve(
                df_points, fit_method_Reg, forecast_length, hidden_layers, activation, max_iter,
                mask_curve_only=mask_curve_only, original_image_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            )

            # --- 5. Fourier globale? ---
            if approx_fourier and x_fit is not None and y_fit is not None:
                x_fourier, y_fourier, fourier_params = approximate_curve_with_fourier(
                    x_fit, y_fit, n_harmonics=fourier_approx_harmonics
                )
                y_fourier_short = np.interp(df_points['X'], x_fourier, y_fourier)
            else:
                y_fourier_short = None

            # --- 6. Cluster? Colori? (e Fourier su cluster) ---
            cluster_label_col = None
            cluster_color_col = None
            y_fourier_clusters_cols = dict()
            if fit_method_Clust in ("Closed path", "Cluster Color"):
                xclust_list, yclust_list, _ = fit_curve(
                    df_points, fit_method_Clust, forecast_length, hidden_layers, activation, max_iter,
                    mask_curve_only=mask_curve_only, original_image_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                )
                # Qui: per ogni cluster, assegna label/color e Fourier!
                cluster_label_col = []
                cluster_color_col = []
                for xi, yi in zip(xclust_list, yclust_list):
                    # Fourier solo se abbastanza punti
                    if approx_fourier and len(xi) > (2 * fourier_approx_harmonics + 2):
                        x_fc, y_fc, _ = approximate_curve_with_fourier(
                            xi, yi, n_harmonics=fourier_approx_harmonics
                        )
                        y_fc_short = np.interp(df_points['X'], x_fc, y_fc)
                    else:
                        y_fc_short = [None] * len(df_points)
                    # Aggiungi come colonna nuova (una per cluster)
                    k = len(y_fourier_clusters_cols)
                    y_fourier_clusters_cols[f"y_approx_fourier_cluster{k}"] = y_fc_short
                    # Assegna label e colore (dummy: numerico, puoi espandere se vuoi i veri colori/nomi)
                    cluster_label_col.extend([k] * len(df_points))
                    cluster_color_col.extend(["#CCCCCC"] * len(df_points)) # puoi mettere il vero colore se disponibile

            # --- 7. Crea DataFrame finale ---
            df_final = pd.DataFrame({
                "N": np.arange(1, len(df_points) + 1),
                "X": df_points["X"].values,
                "Y": df_points["Y"].values,
                "y_fit_primary": y_fit[:len(df_points)] if y_fit is not None else np.nan,
                "image": image_name,
            })
            # Fourier globale
            if y_fourier_short is not None:
                df_final["y_approx_fourier"] = y_fourier_short
            # Cluster label/color
            if cluster_label_col is not None:
                df_final["cluster_label"] = cluster_label_col[:len(df_final)]
            if cluster_color_col is not None:
                df_final["color_cluster"] = cluster_color_col[:len(df_final)]
            # Fourier per ogni cluster (se presenti)
            for colname, yvals in y_fourier_clusters_cols.items():
                df_final[colname] = yvals[:len(df_final)]
            # Altri parametri
            for k, v in params.items():
                df_final[k] = v

            # Aggiungi OCR risultato, se vuoi come colonna:
            df_final["ocr_result"] = ocr_result

            return df_final, mask_curve_only


        # --- INIZIO BLOCCO PRINCIPALE STREAMLIT ---
        if uploaded_file is not None:
            # Inizializzazione di session_state se non già presenti (cruciale per DBSCAN/Cluster Color)
            if 'x0_pix' not in st.session_state:
                st.session_state.x0_pix = 0
                st.session_state.x1_pix = 1
                st.session_state.y0_pix = 0
                st.session_state.y1_pix = 1
                st.session_state.x0_val = 0.0
                st.session_state.x1_val = 1.0
                st.session_state.y0_val = 0.0
                st.session_state.y1_val = 1.0
            #do il valore alla variabile per il dataframe
            st.session_state.uploaded_file = uploaded_file    
            # 1. Leggi e prepara le immagini
            # Assicurati che il cursore sia all'inizio
            uploaded_file.seek(0)

            # Leggi i byte e decodifica l'immagine
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is None:
                st.error("Error decoding the image. The file may be invalid or corrupted.")
            else:
                original_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.session_state.original_image_rgb = original_image_rgb    

            col1, col2 = st.columns([1, 1])
            with col1:
                with st.expander("🖼️ Original Image"):        
                    st.image(original_image_rgb, caption="Loaded Original Image (RGB)", use_container_width=True)

                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                st.session_state.gray_image = gray
                # 2. Call process_image and save the important results in session_state
                ocr_result, mask_lines, mask_curve_only, _ = process_image(
                    img_bgr,
                    st.session_state.canny_low,
                    st.session_state.canny_high,
                    st.session_state.hough_thresh,
                    st.session_state.hough_min_length,
                    st.session_state.hough_max_gap,
                    st.session_state.tol_dx,
                    st.session_state.tol_dy,
                    st.session_state.threshold_val,
                    st.session_state.invert_threshold,
                    st.session_state.alpha_contrast,
                    st.session_state.beta_brightness
                )
                st.session_state.mask_curve_only = mask_curve_only        

            with col2:
                with st.expander("🧠 Detected Text:"):
                    st.text(ocr_result)       
                with st.expander("🔲 Image parts excluded from fitting"):
                    st.image(mask_lines, clamp=True)        

            background_image_for_canvas = None
            if 'mask_curve_only' in st.session_state and st.session_state.mask_curve_only is not None:
                current_mask_for_canvas = st.session_state.mask_curve_only
                if current_mask_for_canvas.size > 0 and current_mask_for_canvas.ndim >= 2:
                    try:
                        mask_bgr = cv2.cvtColor(current_mask_for_canvas, cv2.COLOR_GRAY2BGR)
                        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
                        background_image_for_canvas = Image.fromarray(mask_rgb)
                    except Exception as e:
                        st.error(f"Error converting the mask for the canvas: {e}. The canvas will be empty.")
                else:
                    st.warning("The curve mask is empty or malformed in the session state. The canvas will be empty.")
            else:
                st.warning("Curve mask not available in session state. Upload an image and binarize it.")

            # creo l'altezza e larghezza per il canvas di editing
            if 'mask_curve_only' in st.session_state and st.session_state.mask_curve_only is not None:
                # Assicurati che current_mask_for_canvas sia un NumPy array valido qui
                current_mask_for_canvas = st.session_state.mask_curve_only
                if current_mask_for_canvas.ndim >= 2: # Controlla che sia almeno 2D (altezza, larghezza)
                    original_mask_height, original_mask_width = current_mask_for_canvas.shape[:2] # Ottieni altezza e larghezza
                else:
                    original_mask_height, original_mask_width = 0, 0 # Default se non valido
            else:
                original_mask_height, original_mask_width = 0, 0 # Default se non disponibile

            canvas_height = original_mask_height if original_mask_height > 0 else 500
            canvas_width = original_mask_width if original_mask_width > 0 else 500
            
            # Chiamata a st_canvas con try-except
            canvas_result = None
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)", # Colore di riempimento degli oggetti disegnati. In questo caso, con alpha=1, è bianco opaco.
                    stroke_width=stroke_width_value,     # Larghezza del tratto di disegno, proveniente dal tuo slider.
                    # Il colore del tratto. È cruciale che questo si allinei con la tua logica di editing.
                    # Se usi "Penna (Nera)" il tratto sarà nero, altrimenti bianco (per "Penna (Bianca)" o altri).
                    stroke_color="#000000" if editing_mode == "Penna (Nera)" else "#FFFFFF",
                    background_image=background_image_for_canvas, # L'immagine di sfondo su cui disegnerai.
                    update_streamlit=True,               # Aggiorna Streamlit in tempo reale mentre disegni.
                    height=canvas_height,                # Altezza del canvas.
                    width=canvas_width,                  # Larghezza del canvas.
                    # Modalità di disegno: 'transform' per spostare/ridimensionare gli oggetti esistenti,
                    # 'freedraw' per disegnare a mano libera.
                    drawing_mode="transform" if editing_mode == "Nessuno_Pen" else "freedraw",
                    key="canvas",                        # Una chiave univoca per il componente Streamlit.
                )
            except ValueError as e:
                # Gestione specifica di un errore comune con st_canvas se l'immagine di sfondo non è valida.
                if "The truth value of an array with more than one element is ambiguous" in str(e):
                    st.error("A problem occurred with the canvas background image. Please ensure it is valid and try uploading it again.")
                    canvas_result = None # Assicura che canvas_result sia None per evitare errori successivi
                else:
                    # Se l'errore non è quello specifico, rilanciarlo per non nascondere altri problemi.
                    raise e
            except Exception as e:
                # Catch any other unexpected errors that might occur while using the editing tool.
                st.error(f"An unexpected error occurred while using the editing tool: {e}")
                canvas_result = None  # Ensure that canvas_result is None

            # --- PROCESS THE CANVAS RESULT ---
            if canvas_result is not None and canvas_result.image_data is not None:
                drawn_mask = canvas_result.image_data[:, :, 3] > 0  # Boolean mask of interacted pixels (where alpha > 0)

                if 'mask_curve_only' in st.session_state:
                    current_mask = st.session_state.mask_curve_only.copy()  # Always work on a copy

                    if editing_mode == "Pen (White)":
                        current_mask[drawn_mask] = 255  # Set to WHITE on the mask
                        st.info("Pixels removed (set to white) from the mask.")

                    elif editing_mode == "Pen (Black)":
                        current_mask[drawn_mask] = 0  # Set to BLACK on the mask
                        st.info("Pixels added (set to black) to the mask.")

                    st.session_state.mask_curve_only = current_mask  # Save the modified mask

                    st.subheader("Modified Curve Mask")
                    # Convert the mask from grayscale to BGR for st.image
                    st.image(
                        cv2.cvtColor(st.session_state.mask_curve_only, cv2.COLOR_GRAY2BGR),
                        use_container_width=True,
                        caption="Binary mask after editing (Black/White)"
                    )
                else:
                    st.warning("Curve mask not available for editing.")
            st.divider()
            with st.expander("🟢 Image for fit"):

                        st.image(st.session_state.mask_curve_only, clamp=True)        
                    # Il background del canvas sarà sempre la maschera binaria
            # Pass all relevant parameters to extract_and_convert_points
            # AGGIUNGI QUI I NUOVI PARAMETRI DEL FILTRO
            x_pix, y_pix, df = extract_and_convert_points(st.session_state.mask_curve_only, x0_pix, x1_pix, x0_val, x1_val, y0_pix, y1_pix, y0_val, y1_val, st.session_state.enable_data_filter, st.session_state.filter_window_size, st.session_state.iqr_multiplier)
            st.session_state.df = df       
            if df is not None and not df.empty:
                # Sidebar per limiti degli assi
                x_min_data = float(df["X"].min())
                x_max_data = float(df["X"].max())
                y_min_data = float(df["Y"].min())
                y_max_data = float(df["Y"].max())
                
                x_min = x_min_data
                x_max = x_max_data
                y_min = y_min_data
                y_max = y_max_data
                with st.expander("Fitting"):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(df["X"], df["Y"], color='blue', label="Detected Data")

                    equation_params = None
                    x_fit, y_fit = None, None                    # Initialize x_fit, y_fit
                    x_approx_fourier, y_approx_fourier = None, None  # Initialize Fourier variables

                    # Primary regression or clustering fit
                    if st.session_state.fit_method_Reg != "No_Reg" and not st.session_state.Combine_Methods_Bt:
                        x_fit, y_fit, equation_params = fit_curve(
                            df,
                            st.session_state.fit_method_Reg,
                            st.session_state.forecast_length,
                            st.session_state.hidden_layers,
                            st.session_state.activation,
                            st.session_state.max_iter,
                            mask_curve_only=st.session_state.mask_curve_only,
                            original_image_rgb=original_image_rgb
                        )
                        st.session_state.y_fit_primary = y_fit
                        ax.plot(x_fit, y_fit, color='red', label="Regression Trend")

                    elif st.session_state.Combine_Methods_Bt:
                        x_fit, y_fit, equation_params = combine_fits_iteratively(
                            df,
                            list_of_methods_config,
                            st.session_state.forecast_length,
                            st.session_state.hidden_layers,
                            st.session_state.activation,
                            st.session_state.max_iter,
                            mask_curve_only=st.session_state.mask_curve_only,
                            original_image_rgb=original_image_rgb
                        )
                        st.session_state.y_fit_primary = y_fit
                        ax.plot(x_fit, y_fit, color='green', label="Combined Regression Trend")

                    elif st.session_state.fit_method_Clust != "No_Clust" and not st.session_state.Combine_Methods_Bt:
                        x_fit, y_fit, equation_params = fit_curve(
                            df,
                            st.session_state.fit_method_Clust,
                            st.session_state.forecast_length,
                            st.session_state.hidden_layers,
                            st.session_state.activation,
                            st.session_state.max_iter,
                            mask_curve_only=st.session_state.mask_curve_only,
                            original_image_rgb=original_image_rgb
                        )
                        st.session_state.y_fit_primary = y_fit
                        ax.plot(x_fit, y_fit, color='red')

                    # Compute and display cosine similarity if fit exists
                    if x_fit is not None and y_fit is not None:
                        if ('mask_curve_centered' in locals()
                            and mask_curve_centered is not None
                            and mask_curve_centered.ndim >= 2):
                            similarity = cosine_similarity_between_curves(
                                x_fit, y_fit,
                                mask_curve_centered.shape,
                                mask_curve_centered,
                                x0_pix, x1_pix, x0_val, x1_val,
                                y0_pix, y1_pix, y0_val, y1_val
                            )
                            st.metric("Cosine Similarity", f"{similarity:.3f}")
                        else:
                            st.warning(
                                "Cannot calculate Cosine Similarity: centered mask is not available or is malformed."
                            )
                    else:
                        st.info("No fit calculated; Cosine Similarity will not be shown.")

                    # Display equations for primary fits
                    if st.session_state.fit_method_Clust != "No_Clust":
                        write_equation(st.session_state.fit_method_Clust, equation_params)
                    if st.session_state.fit_method_Reg != "No_Reg":
                        write_equation(st.session_state.fit_method_Reg, equation_params)

                    # Fourier Approximation for non-explicit fits
                    non_explicit_methods = [
                        "Spline", "Neural Network", "Random Forest Regression",
                        "Gradient Boosting", "Support Vector Regression",
                        "Closed path", "Cluster Color"
                    ]
                    if (st.session_state.approx_fourier
                        and (st.session_state.fit_method_Clust in non_explicit_methods
                            or st.session_state.fit_method_Reg in non_explicit_methods
                            or st.session_state.Combine_Methods_Bt)):
                        st.markdown("---")
                        st.subheader("Fourier Approximation of Fitted Curve")

                        # Handle multiple clusters
                        if (st.session_state.fit_method_Clust in ["Closed path", "Cluster Color"]
                            and x_fit and y_fit):
                            for i, (xf, yf) in enumerate(zip(x_fit, y_fit)):
                                if xf is not None and yf is not None and len(xf) > 0 and len(yf) > 0:
                                    temp_x, temp_y, fourier_params = approximate_curve_with_fourier(
                                        xf, yf, st.session_state.fourier_approx_harmonics
                                    )
                                    st.session_state.temp_y_approx_fourier[i] = temp_y
                                    if temp_x is not None and temp_y is not None:
                                        ax.plot(
                                            temp_x, temp_y,
                                            linestyle=':',
                                            label=f"Fourier Approx Cluster {i}",
                                            color='purple'
                                        )
                                        write_equation(
                                            "Fourier",
                                            fourier_params,
                                            approx_type=f"Fourier Approx Cluster {i}"
                                        )
                        else:
                            x_approx_fourier, y_approx_fourier, fourier_params = approximate_curve_with_fourier(
                                x_fit, y_fit, st.session_state.fourier_approx_harmonics
                            )
                            st.session_state.y_approx_fourier = y_approx_fourier
                            if x_approx_fourier is not None and y_approx_fourier is not None:
                                ax.plot(
                                    x_approx_fourier, y_approx_fourier,
                                    linestyle='--',
                                    label=f"Fourier Approx ({st.session_state.fourier_approx_harmonics} harm.)"
                                )
                                write_equation("Fourier", fourier_params, approx_type="Fourier Approximation")
                            else:
                                st.info("Cannot compute Fourier approximation for the fitted curve.")

                        ax.legend()
                        ax.set_xlabel("X")
                        ax.set_ylabel("Y")
                        ax.set_title("Curve Recognition with Forecast")
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        plt.tight_layout()

                    # Center plot if requested
                    if center_plot:
                        all_x = df["X"].copy()
                        all_y = df["Y"].copy()

                        # Include cluster fits
                        if (st.session_state.fit_method_Clust in ["Closed path", "Cluster Color"]
                            and x_fit and y_fit):
                            for xf, yf in zip(x_fit, y_fit):
                                if xf is not None and yf is not None and len(xf) and len(yf):
                                    all_x = np.concatenate([all_x, xf])
                                    all_y = np.concatenate([all_y, yf])
                        elif x_fit is not None and y_fit is not None:
                            all_x = np.concatenate([all_x, x_fit])
                            all_y = np.concatenate([all_y, y_fit])

                        if x_approx_fourier is not None and y_approx_fourier is not None:
                            all_x = np.concatenate([all_x, x_approx_fourier])
                            all_y = np.concatenate([all_y, y_approx_fourier])

                        if len(all_x) > 1 and len(all_y) > 1:
                            margin_x = (all_x.max() - all_x.min()) * 0.05
                            margin_y = (all_y.max() - all_y.min()) * 0.05
                            ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
                            ax.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)
                        elif len(all_x) == 1:
                            ax.set_xlim(all_x[0] - 1, all_x[0] + 1)
                            ax.set_ylim(all_y[0] - 1, all_y[0] + 1)

                        st.pyplot(fig)

                    # Ensure temp_y_approx_fourier is initialized
                    if ("temp_y_approx_fourier" not in st.session_state
                        or not isinstance(st.session_state.temp_y_approx_fourier, dict)):
                        st.session_state.temp_y_approx_fourier = {}
                        st.warning("st.session_state.temp_y_approx_fourier was not initialized or not a dict. Reinitialized to {}.")

                st.markdown("---")

                with st.expander("📥 Download DataFrame"):  
                    params = st.session_state.sidebar_params
                    enable_data_filter = params['enable_data_filter']
                    filter_window_size = params['filter_window_size']
                    iqr_multiplier = params['iqr_multiplier']
                    threshold_val = params['threshold_val']
                    invert_threshold = params['invert_threshold']
                    alpha_contrast = params['alpha_contrast']
                    beta_brightness = params['beta_brightness']
                    canny_low = params['canny_low']
                    canny_high = params['canny_high']
                    hough_thresh = params['hough_thresh']
                    hough_min_length = params['hough_min_length']
                    hough_max_gap = params['hough_max_gap']
                    tol_dx = params['tol_dx']
                    tol_dy = params['tol_dy']            

                    center_plot = params['center_plot']
                    x0_pix = params['x0_pix']
                    x1_pix = params['x1_pix']
                    x0_val = params['x0_val']
                    x1_val = params['x1_val']
                    y0_pix = params['y0_pix']
                    y1_pix = params['y1_pix']
                    y0_val = params['y0_val']
                    y1_val = params['y1_val']

                    fit_method_Reg = params['fit_method_Reg']
                    fit_method_Clust = params['fit_method_Clust']
                    N_color_clusters = params['N_color_clusters']
                    window_size = params['window_size']
                    hidden_layers = params['hidden_layers']
                    activation = params['activation']
                    max_iter = params['max_iter']
                    Combine_Methods_Bt = params['Combine_Methods_Bt']
                    # selected_methods_combined = params['selected_methods_combined'] # If uncommented
                    # list_of_methods_config = params['list_of_methods_config'] # If uncommented
                    forecast_length = params['forecast_length']
                    approx_fourier = params['approx_fourier']

                    dbscan_min_samples = params['dbscan_min_samples']
                    dbscan_eps = params['dbscan_eps']
                    pixel_proximity_threshold = params['pixel_proximity_threshold']
                    perimeter_offset_radius = params['perimeter_offset_radius']
                    perimeter_smoothing_sigma = params['perimeter_smoothing_sigma']
                    path_fit_type = params['path_fit_type']



                    if not st.session_state.df.empty and st.session_state.uploaded_file is not None:

                        # --- DEBUG: Visualizza lo stato delle variabili prima dell'uso ---
                        # Questi messaggi sono utili per capire il contenuto delle variabili
                                        
                        st.info(f"Stato di st.session_state.cluster_labels all'inizio del download: {st.session_state.cluster_labels.shape if isinstance(st.session_state.cluster_labels, np.ndarray) else type(st.session_state.cluster_labels)}")
                        st.info(f"Stato di st.session_state.temp_y_approx_fourier all'inizio del download: {len(st.session_state.temp_y_approx_fourier)} cluster trovati.")
                        if st.session_state.temp_y_approx_fourier:
                            for k, v in st.session_state.temp_y_approx_fourier.items():
                                st.info(f"  Cluster {k}: Dati Fourier di tipo {type(v)}, forma {v.shape if isinstance(v, np.ndarray) else 'N/A'}")

                        # --- Assicurati che queste variabili siano numpy array e appiattisci se necessario ---
                        # Controlli di tipo e dimensione rafforzati
                        if isinstance(st.session_state.y_fit_primary, list):
                            st.session_state.y_fit_primary = np.array(st.session_state.y_fit_primary)
                        if st.session_state.y_fit_primary.ndim > 1:
                            st.session_state.y_fit_primary = st.session_state.y_fit_primary.flatten()
                        
                        if isinstance(st.session_state.y_approx_fourier, list):
                            st.session_state.y_approx_fourier = np.array(st.session_state.y_approx_fourier)
                        if hasattr(st.session_state.y_approx_fourier, "ndim") and st.session_state.y_approx_fourier is not None:
                            if st.session_state.y_approx_fourier.ndim > 1:
                                st.session_state.y_approx_fourier = st.session_state.y_approx_fourier.flatten()

                        if isinstance(st.session_state.x_extended, list):
                            st.session_state.x_extended = np.array(st.session_state.x_extended)
                        if st.session_state.x_extended.ndim > 1:
                            st.session_state.x_extended = st.session_state.x_extended.flatten()

                        # >>> Gestione di cluster_labels, anche se è un set <<<
                        if isinstance(st.session_state.cluster_labels, set):
                            st.session_state.cluster_labels = np.array(list(st.session_state.cluster_labels))
                            st.warning("Convertito st.session_state.cluster_labels da set a numpy array.")
                        elif isinstance(st.session_state.cluster_labels, list):
                            st.session_state.cluster_labels = np.array(st.session_state.cluster_labels)
                        if isinstance(st.session_state.cluster_labels, np.ndarray) and st.session_state.cluster_labels.ndim > 1:
                            # Your existing logic for multi-dimensional cluster_labels goes here
                            # Example: Accessing elements like cluster_labels[:, 0] or cluster_labels[0, 1]
                            # ...
                            st.write("Cluster labels are a multi-dimensional NumPy array.")
                        elif isinstance(st.session_state.cluster_labels, np.ndarray) and st.session_state.cluster_labels.ndim == 1:
                            st.write("Cluster labels are a 1D NumPy array (likely actual labels).")
                            # Handle the 1D case, for example, by counting unique labels or just passing them
                            # For instance, if you expect coordinates, this would be where you catch the error
                            # and maybe display a warning or fallback.
                        else:
                            # This covers cases where it's not a NumPy array or is an empty array that needs special handling
                            st.write("Cluster labels are not a suitable NumPy array yet, or are empty.")
                            # You might want to skip the processing that requires cluster_labels at this point
                            # or provide a default empty result.
                            pass # Or log a message, show a placeholder, etc.
                        x_val = np.atleast_1d(st.session_state.x_extended)
                        df_final_download = pd.DataFrame({'x_extended': x_val})
                        df_final_download.insert(0, 'N', np.arange(1, len(df_final_download) + 1))
                    # DEBUG per colore_cluster (Versione 1: colori per ogni punto)
                        if 'colore_cluster' in st.session_state and st.session_state.colore_cluster is not None:
                            if isinstance(st.session_state.colore_cluster, (list, tuple)):
                                # Converti a numpy array e assicurati che sia Nx3 se è una lista di tuple/liste RGB
                                try:
                                    st.session_state.colore_cluster = np.array(st.session_state.colore_cluster)
                                    if st.session_state.colore_cluster.ndim == 1 and st.session_state.colore_cluster.size > 0 and isinstance(st.session_state.colore_cluster[0], (tuple, list)):
                                        # Se è un array di tuple/liste (come [(R,G,B), (R,G,B), ...])
                                        st.session_state.colore_cluster = np.vstack(st.session_state.colore_cluster)
                                    st.info(f"Converted st.session_state.colore_cluster to a NumPy array. Shape: {st.session_state.colore_cluster.shape}.")
                                except ValueError:
                                    st.warning("Cannot convert st.session_state.colore_cluster to an N×3 NumPy array. Ensure its elements are 3-tuples or 3-lists.")
                                    st.session_state.colore_cluster = np.array([])  # Clear it to avoid further errors

                                if isinstance(st.session_state.colore_cluster, np.ndarray) and st.session_state.colore_cluster.ndim == 1:
                                    # If it’s a 1D array but should be N×3, that’s a problem.
                                    # It might mean there’s only one color for all points.
                                    st.warning("st.session_state.colore_cluster is a 1D array. It may not be in [N, 3] format.")
                                    # Decide how to handle it: fill with NaN or treat as a single color.
                                    # For now, just warn and attempt to treat it as N×3 later if possible.

                                else:
                                    st.session_state.colore_cluster = np.array([])
                                    st.warning("st.session_state.colore_cluster is not present or is None. Treated as empty for RGB channels.")

                                # --- ADD COLOR_CLUSTER COLUMNS (R, G, B) ---
                                if (
                                    st.session_state.colore_cluster.size > 0
                                    and st.session_state.colore_cluster.ndim == 2
                                    and st.session_state.colore_cluster.shape[1] == 3
                                ):
                                    # If it’s an N×3 array, extract each channel
                                    min_len_colors = min(len(df_final_download), st.session_state.colore_cluster.shape[0])
                                    df_final_download['colore_cluster_R'] = np.full(len(df_final_download), np.nan)
                                    df_final_download['colore_cluster_G'] = np.full(len(df_final_download), np.nan)
                                    df_final_download['colore_cluster_B'] = np.full(len(df_final_download), np.nan)

                                    df_final_download.loc[:min_len_colors-1, 'colore_cluster_R'] = st.session_state.colore_cluster[:min_len_colors, 0]
                                    df_final_download.loc[:min_len_colors-1, 'colore_cluster_G'] = st.session_state.colore_cluster[:min_len_colors, 1]
                                    df_final_download.loc[:min_len_colors-1, 'colore_cluster_B'] = st.session_state.colore_cluster[:min_len_colors, 2]
                                else:
                                    st.warning("st.session_state.colore_cluster is not in the expected Nx3 format or is empty. R, G, B columns will be NaN.")
                                    df_final_download['colore_cluster_R'] = np.full(len(df_final_download), np.nan)
                                    df_final_download['colore_cluster_G'] = np.full(len(df_final_download), np.nan)
                                    df_final_download['colore_cluster_B'] = np.full(len(df_final_download), np.nan)

                                # --- Add primary fit Y values (based on x_extended) ---
                                df_final_download['y_fit_primary'] = np.full(len(df_final_download), np.nan)
                                if st.session_state.y_fit_primary.size > 0:
                                    y_val = np.atleast_1d(st.session_state.y_fit_primary)
                                    min_len = min(len(df_final_download), len(y_val))
                                    df_final_download.loc[:min_len-1, 'y_fit_primary'] = y_val[:min_len]
                                else:
                                    st.warning("st.session_state.y_fit_primary is empty. 'y_fit_primary' column filled with NaN.")

                                # --- Add Fourier approximation Y values (based on x_extended) ---
                                df_final_download['y_approx_fourier'] = np.full(len(df_final_download), np.nan)
                                if st.session_state.y_approx_fourier.size > 0:
                                    y_val = np.atleast_1d(st.session_state.y_approx_fourier)
                                    min_len = min(len(df_final_download), len(y_val))
                                    df_final_download.loc[:min_len-1, 'y_approx_fourier'] = y_val[:min_len]
                                else:
                                    st.warning("st.session_state.y_approx_fourier is empty. 'y_approx_fourier' column filled with NaN.")

                                # --- Handle Y_Real_Original and cluster_labels ---
                                df_final_download['Y_Real_Original'] = np.nan
                                df_final_download['cluster_labels'] = np.full(len(df_final_download), np.nan)

                                if isinstance(st.session_state.cluster_labels, np.ndarray) and st.session_state.cluster_labels.size > 0:
                                    y_val = np.atleast_1d(st.session_state.cluster_labels)
                                    min_len = min(len(df_final_download), len(y_val))
                                    df_final_download.loc[:min_len-1, 'cluster_labels'] = y_val[:min_len]
                                elif isinstance(st.session_state.cluster_labels, np.ndarray) and st.session_state.cluster_labels.size == 0:
                                    st.write("cluster_labels is an empty NumPy array. No clusters processed.")
                                else:
                                    st.error(f"Error: st.session_state.cluster_labels is not a NumPy array. Current type: {type(st.session_state.cluster_labels)}")

                                # --- Add Fourier transforms for each cluster (temp_y_approx_fourier) ---
                                if st.session_state.temp_y_approx_fourier:  # Check if the dict has entries
                                    for cluster_id, fourier_data in st.session_state.temp_y_approx_fourier.items():
                                        col_name = f'Fourier_Cluster_{cluster_id}'

                                        # Ensure fourier_data is not None before proceeding
                                        if fourier_data is None:
                                            st.warning(f"Fourier data for Cluster {cluster_id} is None. Skipping this column.")
                                            continue

                                        # Convert to NumPy array and flatten if needed
                                        if isinstance(fourier_data, list):
                                            fourier_data = np.array(fourier_data)
                                            st.info(f"Converted Fourier data for Cluster {cluster_id} from list to NumPy array.")

                                        if fourier_data.ndim > 1:  # Safe to check after conversion
                                            fourier_data = fourier_data.flatten()
                                            st.info(f"Flattened Fourier data for Cluster {cluster_id} to 1D.")

                                        df_final_download[col_name] = np.full(len(df_final_download), np.nan)

                                        if fourier_data.size > 0:  # Proper check for NumPy arrays
                                            min_len_fourier = min(len(df_final_download), len(fourier_data))
                                            df_final_download.loc[:min_len_fourier-1, col_name] = fourier_data[:min_len_fourier]
                                        else:
                                            st.warning(f"Fourier data for Cluster {cluster_id} is empty. Column '{col_name}' filled with NaN.")
                                else:
                                    st.info("No Fourier data in st.session_state.temp_y_approx_fourier to add.")

                    else: # Se x_extended non è disponibile, usa solo i punti reali come base
                        df_final_download = st.session_state.df.copy()
                        df_final_download.insert(0, 'N', np.arange(1, len(df_final_download) + 1))
                        df_final_download.rename(columns={'X': 'X_Real_Original', 'Y': 'Y_Real_Original'}, inplace=True)
                        
                        df_final_download['y_approx_fourier'] = np.nan
                        df_final_download['y_fit_primary'] = np.nan

                        # --- DEBUG: Visualize variable states before raw DataFrame download ---
                        st.info(
                            f"State of st.session_state.cluster_labels at download start (raw DF): "
                            f"{st.session_state.cluster_labels.shape if isinstance(st.session_state.cluster_labels, np.ndarray) else type(st.session_state.cluster_labels)}"
                        )
                        st.info(
                            f"State of st.session_state.temp_y_approx_fourier at download start (raw DF): "
                            f"{len(st.session_state.temp_y_approx_fourier)} clusters found."
                        )
                        if st.session_state.temp_y_approx_fourier:
                            for k, v in st.session_state.temp_y_approx_fourier.items():
                                st.info(
                                    f"  Cluster {k} (raw DF): Fourier data type {type(v)}, "
                                    f"shape {v.shape if isinstance(v, np.ndarray) else 'N/A'}"
                                )

                        # >>> Handle cluster_labels even if it’s a set (raw DF case) <<<
                        if isinstance(st.session_state.cluster_labels, set):
                            st.session_state.cluster_labels = np.array(list(st.session_state.cluster_labels))
                            st.warning("Converted st.session_state.cluster_labels from set to NumPy array (raw DF).")
                        elif isinstance(st.session_state.cluster_labels, list):
                            st.session_state.cluster_labels = np.array(st.session_state.cluster_labels)

                        if st.session_state.cluster_labels.ndim > 1:
                            st.session_state.cluster_labels = st.session_state.cluster_labels.flatten()
                            st.warning("Flattened st.session_state.cluster_labels to 1D (raw DF).")

                        df_final_download['cluster_labels'] = np.full(len(df_final_download), np.nan)
                        if st.session_state.cluster_labels.size > 0:
                            y_val = np.atleast_1d(st.session_state.cluster_labels)
                            min_len = min(len(df_final_download), len(y_val))
                            df_final_download.loc[:min_len-1, 'cluster_labels'] = y_val[:min_len]
                        else:
                            st.warning("st.session_state.cluster_labels is empty (raw DF). 'cluster_labels' column filled with NaN.")

                        # --- ADD FOURIER TRANSFORMS FOR EACH CLUSTER (temp_y_approx_fourier) FOR RAW POINTS ---
                        if st.session_state.temp_y_approx_fourier:
                            for cid, fourier_data in st.session_state.temp_y_approx_fourier.items():
                                col_name = f'Fourier_Cluster_{cid}'
                                if isinstance(fourier_data, list):
                                    fourier_data = np.array(fourier_data)
                                    st.info(f"Converted Fourier data for Cluster {cid} from list to NumPy array (raw DF).")
                                if fourier_data.ndim > 1:
                                    fourier_data = fourier_data.flatten()
                                    st.info(f"Flattened Fourier data for Cluster {cid} to 1D (raw DF).")

                                df_final_download[col_name] = np.full(len(df_final_download), np.nan)
                                if fourier_data.size > 0:
                                    min_len_fourier = min(len(df_final_download), len(fourier_data))
                                    df_final_download.loc[:min_len_fourier-1, col_name] = fourier_data[:min_len_fourier]
                                else:
                                    st.warning(
                                        f"Fourier data for Cluster {cid} is empty (raw DF). "
                                        f"Column '{col_name}' filled with NaN."
                                    )
                        else:
                            st.info("No Fourier data in st.session_state.temp_y_approx_fourier to add (raw DF).")


                    # --- AGGIUNTA DEI PARAMETRI SCALARI DALLA SIDEBAR ---
                    # Questi parametri vengono aggiunti al df_final_download
                    # Utilizziamo .at[0, 'NomeColonna'] per impostare il valore solo nella prima riga
                    # e poi fillna per riempire il resto, se la colonna non esiste viene creata con nan

                    # Editing Immagine
                    
                    df_final_download['Param_Enable_Data_Filter'] = enable_data_filter
                    df_final_download['Param_Filter_Window_Size'] = filter_window_size
                    df_final_download['Param_IQR_Multiplier'] = iqr_multiplier
                    df_final_download['Param_Threshold_Value'] = threshold_val
                    df_final_download['Param_Invert_Threshold'] = invert_threshold
                    df_final_download['Param_Alpha_Contrast'] = alpha_contrast
                    df_final_download['Param_Beta_Brightness'] = beta_brightness
                    df_final_download['Param_Canny_Low'] = canny_low
                    df_final_download['Param_Canny_High'] = canny_high
                    df_final_download['Param_Hough_Threshold'] = hough_thresh
                    df_final_download['Param_Hough_Min_Length'] = hough_min_length
                    df_final_download['Param_Hough_Max_Gap'] = hough_max_gap
                    df_final_download['Param_Tol_DX'] = tol_dx
                    df_final_download['Param_Tol_DY'] = tol_dy            

                    # Editing Assi
                    df_final_download['Param_Center_Plot'] = center_plot
                    df_final_download['Param_X0_Pix'] = x0_pix
                    df_final_download['Param_X1_Pix'] = x1_pix
                    df_final_download['Param_X0_Val'] = x0_val
                    df_final_download['Param_X1_Val'] = x1_val
                    df_final_download['Param_Y0_Pix'] = y0_pix
                    df_final_download['Param_Y1_Pix'] = y1_pix
                    df_final_download['Param_Y0_Val'] = y0_val
                    df_final_download['Param_Y1_Val'] = y1_val

                    # Metodi Matematici (Regressione e Clustering)
                    df_final_download['Param_Fit_Method_Reg'] = fit_method_Reg
                    df_final_download['Param_Fit_Method_Clust'] = fit_method_Clust
                    df_final_download['Param_N_Color_Clusters'] = N_color_clusters
                    df_final_download['Param_Window_Size_MA'] = window_size # Per Movie means
                    df_final_download['Param_NN_Hidden_Layers'] = hidden_layers
                    df_final_download['Param_NN_Activation'] = activation
                    df_final_download['Param_NN_Max_Iter'] = max_iter
                    df_final_download['Param_Combine_Methods'] = Combine_Methods_Bt
                    # Per i metodi combinati, potresti voler serializzare la lista_of_methods_config
                    # in una stringa o un JSON per salvarla in una singola cella, o gestirla diversamente.
                    # Per semplicità qui aggiungo solo i nomi dei metodi selezionati.            
                    df_final_download['Param_Forecast_Length'] = forecast_length
                    df_final_download['Param_Approx_Fourier'] = approx_fourier

                    # Parametri Clustering Specifici
                    df_final_download['Param_DBSCAN_Min_Samples'] = dbscan_min_samples
                    df_final_download['Param_DBSCAN_EPS'] = dbscan_eps
                    df_final_download['Param_Pixel_Proximity_Threshold'] = pixel_proximity_threshold
                    df_final_download['Param_Perimeter_Offset_Radius'] = perimeter_offset_radius
                    df_final_download['Param_Perimeter_Smoothing_Sigma'] = perimeter_smoothing_sigma
                    df_final_download['Param_Path_Fit_Type'] = path_fit_type
            
                    
                    # The image name will be included in the file name to avoid repetition in the CSV
                    image_base_name = st.session_state.uploaded_file.name.replace('.', '_')
                    # Display the Analysis Data Table converted from Excel in Streamlit
                    st.subheader("Analysis Data Table", help="Scroll through the DataFrame")
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Create a copy of the DataFrame
                        df_excel = df_final_download.copy()

                        # Clear parameter columns (except the first row)
                        for col in df_excel.columns:
                            if col.startswith("Param_"):
                                df_excel.loc[1:, col] = ""  # or np.nan

                        # Write the Data sheet
                        df_excel.to_excel(writer, sheet_name="Data", index=False)

                        # Write the Parameters sheet from the sidebar
                        params = st.session_state.sidebar_params
                        params_df = pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
                        params_df.to_excel(writer, sheet_name="Parameters", index=False)

                    # End of 'with' block
                    output.seek(0)
                    st.dataframe(df_excel)
                    # Download button for the Excel file
                    st.download_button(
                        label="Download Excel Data",
                        data=output,
                        file_name="output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    # Load the Excel file from BytesIO (without saving to disk)
                    output.seek(0)
                    df_from_excel = pd.read_excel(output, sheet_name="Data")

                    # Convert to CSV
                    csv_from_excel = df_from_excel.to_csv(index=False).encode('utf-8')

                    # Download button for the CSV derived from Excel
                    st.download_button(
                        label="Download CSV Data",
                        data=csv_from_excel,
                        file_name=f"DF_{image_base_name}.csv",
                        mime="text/csv"
                    )

            else:  # If st.session_state.df is empty or no file has been uploaded
                st.info("Upload an image and process it to extract points and enable database download.")

        # This final `else` is important for the structure of your Streamlit script
        else:
            st.info("Upload an image to proceed.")


    with st.sidebar:
        st.markdown("---")
        st.markdown("<h2 style='color: #EB7323;'>Module B</h2>", unsafe_allow_html=True)            
        with st.expander("⚙️ Classifier Parameters"):                
            st.subheader(
                "Training Model",
                help=(
                    "Try Logistic Regression or Neural Network; meanwhile, view the scatter plot, "
                    "and vary the number of neurons and layers (e.g. 8 or 8,9,8 for 3 layers with "
                    "8, 9, and 8 neurons respectively)"
                )
            )
            model_choice = st.selectbox(
                "Choose a Model:",
                (
                    "Support Vector Machine (SVC)",
                    "Logistic Regression",
                    "Random Forest",
                    "Neural Network (MLP)"
                ),
                key="model_selector"
            )

            fourier_harmonics_slider = st.slider(
                "Number of Fourier Harmonics (per Feature)",
                1, 20, 5, 1,
                help="Determines how many Fourier coefficients to use for feature extraction."
            )

            # Specific controls for SVC
            if model_choice == "Support Vector Machine (SVC)":
                svc_c = st.slider(
                    "SVC - Parameter C",
                    0.1, 10.0, 1.0, 0.1,
                    help="Penalty for classification errors."
                )
                svc_kernel = st.selectbox(
                    "SVC - Kernel",
                    ("linear", "rbf", "poly"),
                    key="svc_kernel"
                )
                st.session_state.model_params = {'C': svc_c, 'kernel': svc_kernel}

            elif model_choice == "Logistic Regression":
                lr_c = st.slider(
                    "LR - Parameter C",
                    0.1, 10.0, 1.0, 0.1,
                    help="Inverse of regularization strength."
                )
                st.session_state.model_params = {'C': lr_c}

            elif model_choice == "Random Forest":
                rf_estimators = st.slider(
                    "RF - Number of Estimators",
                    10, 200, 100, 10,
                    help="Number of trees in the forest."
                )
                rf_max_depth = st.slider(
                    "RF - Max Depth",
                    2, 20, 10, 1,
                    help="Maximum depth of the trees."
                )
                st.session_state.model_params = {
                    'n_estimators': rf_estimators,
                    'max_depth': rf_max_depth
                }

            elif model_choice == "Neural Network (MLP)":
                mlp_hidden_layer_sizes = st.text_input(
                    "MLP - Hidden Layer Sizes (e.g., 100 or 50,30)",
                    value="100",
                    help="Number of neurons in hidden layers, comma-separated."
                )
                mlp_hidden_layer_sizes = tuple(
                    int(x.strip())
                    for x in mlp_hidden_layer_sizes.split(",")
                    if x.strip().isdigit()
                )
                mlp_activation = st.selectbox(
                    "MLP - Activation Function",
                    ("relu", "tanh", "logistic"),
                    index=0
                )
                mlp_alpha = st.slider(
                    "MLP - Alpha Parameter (L2 regularization)",
                    0.0001, 0.01, 0.001, 0.0001
                )
                mlp_solver = st.selectbox(
                    "MLP - Solver",
                    ("adam", "lbfgs", "sgd"),
                    index=0
                )
                st.session_state.model_params = {
                    'hidden_layer_sizes': mlp_hidden_layer_sizes,
                    'activation': mlp_activation,
                    'alpha': mlp_alpha,
                    'solver': mlp_solver
                }

            # This slider is for the "manual feedback" phase, not the trained model.
            similarity_threshold_slider = st.slider(
                "Similarity Threshold (for initial feedback)",
                0.0, 1.0, st.session_state.similarity_threshold, 0.01,
                help=(
                    "Threshold used to classify curves as 'same or different' "
                    "during manual feedback."
                )
            )
            st.session_state.similarity_threshold = similarity_threshold_slider

            st.subheader("Variance Analysis on Classified Points")
            show_var_analysis = st.checkbox(
                "Show class/outlier variance analysis",
                value=True,
                key="show_var_analisi"
            )
            var_outlier_zscore = st.number_input(
                "Outlier z-score threshold",
                1.0, 5.0, 2.5, 0.1,
                key="var_outlier_zscore"
            )
            # Checkbox to exclude outliers
            exclude_outlier = st.checkbox(
                "Exclude Outliers (by model and z-score)",
                value=False,
                key="exclude_stat_outlier"
            )
            st.session_state.exclude_outlier = exclude_outlier


    with tab2:
        st.markdown(tab_labels[1], unsafe_allow_html=True)
        def get_model(model_choice, params):
            if model_choice == "Support Vector Machine (SVC)":
                return SVC(**params, probability=True)
            elif model_choice == "Logistic Regression":
                return LogisticRegression(**params)
            elif model_choice == "Random Forest":
                return RandomForestClassifier(**params)
            elif model_choice == "Neural Network (MLP)":
                return MLPClassifier(**params)
            else:
                return None

        def main_effects_with_var_plot(df, feature_names, class_labels=["Different (0)", "Equal (1)"]):
            n = len(feature_names)
            fig, axs = plt.subplots(n, 1, figsize=(3 + 2.4*len(class_labels), 4*n))

            if n == 1:
                axs = [axs]

            for i, feat in enumerate(feature_names):
                means = df.groupby("label")[feat].mean()
                stds  = df.groupby("label")[feat].std()

                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_title(f"Target Plot: {feat}")

                # SCALA ADATTIVA: la std più grande -> raggio max visibile (es 0.22)
                max_std = stds.max() if stds.max() > 0 else 1
                max_radius = 0.22
                scale = max_radius / max_std

                xpos_arr = np.linspace(0.25, 0.75, len(means))
                for idx, lbl in enumerate(means.index):
                    xpos = xpos_arr[idx]
                    ypos = 0.5

                    radius = scale * stds[lbl]
                    color = "#E57373" if lbl==0 else "#64B5F6"

                    circle = plt.Circle((xpos, ypos), radius, color=color, alpha=0.3, zorder=1)
                    axs[i].add_patch(circle)
                    axs[i].plot(xpos, ypos, 'o', color="#C62828" if lbl==0 else "#1565C0", markersize=18, zorder=2)
                    axs[i].text(xpos, ypos-radius-0.09, 
                                f"{class_labels[lbl]}\nμ={means[lbl]:.2f}\nσ={stds[lbl]:.2f}", 
                                ha='center', va='top', fontsize=11)

                axs[i].set_xlim(0, 1)
                axs[i].set_ylim(0, 1)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


            # --- FUNZIONE ANALISI VARIANZA SU FEATURE CLASSIFICATE ---
        def analisi_varianza_feature_classificati(feature_list, label_list, file_names, feature_names, zscore_thr=2.5):
                """
                Calcola varianza, media, std e z-score per ciascuna feature delle classi (0/1) e segnala outlier statistici.
                Restituisce:
                    - df_var (statistiche per feature e classe)
                    - outlier_df (tabella dei punti che sono outlier per almeno una feature)
                """
                import pandas as pd
                import numpy as np
                if not feature_list or not label_list or len(feature_list) != len(label_list):
                    return None, None

                X = np.array(feature_list)  # shape (N, n_feat)
                y = np.array(label_list)
                file_names = list(file_names)
                # … codice che prepara X e feature_names …

                # allineo feature_names a X.shape[1]
                n_features = X.shape[1]
                if len(feature_names) < n_features:
                    feature_names = feature_names + [f"F{i}" for i in range(len(feature_names), n_features)]
                feature_names = feature_names[:n_features]

                # riga 2313
                df = pd.DataFrame(X, columns=feature_names)
                df["label"] = y
                df["file_name"] = file_names

                # Statistiche per ciascuna classe
                stats = []
                for class_val in sorted(df["label"].unique()):
                    sub = df[df["label"] == class_val]
                    stats.append(sub[feature_names].agg(["mean","std","var"]).assign(label=class_val))
                df_var = pd.concat(stats, keys=[f"label_{v}" for v in sorted(df["label"].unique())])

                # Calcolo z-score rispetto ai punti della propria classe
                zscores = []
                for class_val in sorted(df["label"].unique()):
                    sub = df[df["label"] == class_val]
                    means = sub[feature_names].mean()
                    stds = sub[feature_names].std().replace(0, 1e-8)  # evita div zero
                    z = ((sub[feature_names] - means) / stds).abs()
                    z["file_name"] = sub["file_name"]
                    z["label"] = class_val
                    zscores.append(z)
                z_all = pd.concat(zscores, ignore_index=True)
                # Outlier: almeno una feature sopra la soglia
                outlier_mask = (z_all[feature_names] > zscore_thr).any(axis=1)
                outlier_df = z_all[outlier_mask][["file_name","label"] + feature_names]

                return df_var, outlier_df, df

        def plot_decision_boundary(model, X, y):
            # Outlier statistici
            outlier_idx = st.session_state.get('outlier_idx', [])
            if st.session_state.get("exclude_outlier", False) and outlier_idx:
                mask = np.array([i not in outlier_idx for i in range(len(y))])
                X_plot = X[mask]
                y_plot = np.array(y)[mask]
            else:
                X_plot = X
                y_plot = np.array(y)

            # --- Limiti assi sempre sui dati COMPLETI ---
            x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
            y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(grid).reshape(xx.shape)

            plt.figure(figsize=(8,6))
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
            scatter = plt.scatter(X_plot[:,0], X_plot[:,1], c=y_plot, cmap='coolwarm', edgecolors='k')
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.title("Decision Boundary con PCA")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.colorbar(scatter)
            st.pyplot(plt.gcf())
            plt.close()
            # --- 2) Outlier detection ---
            y_pred = model.predict(X)
            outlier_indices = np.where(y_pred != y)[0]
            st.session_state.outlier_indices = outlier_indices.tolist()

            # --- 3) Tabella feedback ---
            file_names = st.session_state.get("feedback_file_names", [])
            y_arr      = st.session_state.get("feedback_labels", [])  # <- Cambiato qui!
            outlier_indices    = st.session_state.get("outlier_indices", [])

            if not file_names or len(file_names) != len(y_arr):
                st.warning("⚠ Cannot build the table: feedback_file_names and feedback_labels must have the same length.")
                return

            df_summary = pd.DataFrame({
                "file_name":   file_names,
                "class_label": y_arr,
                "is_outlier":  [i in outlier_indices for i in range(len(y_arr))]
            })
            
            df_summary["class_name"] = df_summary["class_label"].map({0: "Different", 1: "Equal"})
            # --- TABELLONE FINALE (AGGIORNATO SE SI ESCLUDONO OUTLIER) ---
            file_names = st.session_state.get("feedback_file_names", [])
            y_arr      = st.session_state.get("feedback_labels", [])
            outlier_idx    = st.session_state.get("outlier_indices", [])

            if not file_names or len(file_names) != len(y_arr):
                st.warning("⚠ Cannot build the table: feedback_file_names and feedback_labels must have the same length.")
            else:
                # Prendi la variabile outlier_idx da session_state (NON crearla locale!)
                outlier_idx = st.session_state.get('outlier_idx', [])

                # Filtra in base alla checkbox
                if st.session_state.exclude_outlier and outlier_idx:
                    mask = [i not in outlier_idx for i in range(len(y_arr))]
                else:
                    mask = [True]*len(y_arr)
                df_summary = pd.DataFrame({
                    "file_name":   np.array(file_names)[mask],
                    "class_label": np.array(y_arr)[mask],
                    "is_outlier":  [i in outlier_indices for i in range(len(y_arr)) if mask[i]]
                })        
                df_summary["class_name"] = df_summary["class_label"].map({0: "Different", 1: "Equal"})
            
            st.subheader("🧮🥏 Classi e Outlier secondo il modello di classificazione")
            st.info("""
                Mostra la lista di tutti i punti caricati, indicando a quale classe appartengono secondo le etichette (ad es. “Different” o “Equal”).
                Per ogni punto è anche indicato se il modello di classificazione lo considera “outlier” (cioè se la previsione del modello non coincide con la classe assegnata).
                Serve a identificare rapidamente i casi problematici per il modello.
                """)

            st.dataframe(df_summary)
        # Recupera le features (array NxD), labels e nomi file dalle liste di feedbackfeatures_arr = np.array(st.session_state.get("feedback_features", []))  # NxD
            features_arr = np.array(st.session_state.get("feedback_features", []))  # NxD
            labels_arr   = np.array(st.session_state.get("feedback_labels", []))
            file_names   = st.session_state.get("feedback_file_names", [])
            outlier_indices = []  # <- sempre definita!

            if len(features_arr) > 0 and len(labels_arr) == len(features_arr):
                z_thr = st.session_state.get("var_outlier_zscore", 2.5)
                st.subheader("🧮🥏 Classes and Outliers according to the classification model")
                st.info("""
                    Shows the list of all loaded points, indicating which class they belong to according to the labels (e.g., “Different” or “Equal”).
                    For each point, it also indicates whether the classification model considers it an “outlier” (i.e., if the model’s prediction does not match the assigned class).
                    This helps quickly identify problematic cases for the model.
                    """)

                # 1. Individua gli outlier (salvati come indici)
            # outlier_indices = []
                outlier_descr = []
                for lbl in [0, 1]:
                    mask = (labels_arr == lbl)
                    if mask.sum() > 0:
                        class_feat = features_arr[mask]
                        z_scores = np.abs(zscore(class_feat, axis=0, nan_policy='omit'))
                        max_z = np.nanmax(z_scores, axis=1)
                        idx_this_class = np.where(mask)[0]
                        class_outlier_idx = idx_this_class[max_z > z_thr]
                        outlier_indices.extend(class_outlier_idx.tolist())
                        for i in class_outlier_idx:
                            outlier_descr.append({
                                "file_name": file_names[i],
                                "label": int(labels_arr[i]),
                                "max_abs_z": float(max_z[i - idx_this_class[0]])
                            })

                if outlier_descr:
                    df_out = pd.DataFrame(outlier_descr)
                    st.warning(f"**Outliers found ({len(df_out)}) among the labeled points.** You can adjust the threshold in the sidebar.")
                    st.dataframe(df_out)
                else:
                    st.info("No statistical outliers found among the labeled data (with the current threshold).")

                if len(features_arr) > 0 and len(labels_arr) == len(features_arr):
                    # ... (already done: calculation of outlier_idx etc)


                    # --- GRAFICI GAUSSIANI (parola chiave: PLOT_GAUSSIANI) ---
                    # Scegli la feature principale: media su tutte, oppure la 1a/2a feature
                    # Qui prendo la prima colonna delle feature come esempio
                    feature_idx = 0  # oppure scegli tu la feature da visualizzare (es. con uno selectbox)
                    feat_name = st.session_state.all_feature_names[feature_idx] if "all_feature_names" in st.session_state else f"Feature {feature_idx+1}"

                    for lbl, color, name in zip([0,1], ["#E57373", "#64B5F6"], ["Different", "Equal"]):
                        class_feat = features_arr[labels_arr == lbl, feature_idx]
                        if len(class_feat) == 0:
                            continue  # Nessun dato per questa classe

                        mu, sigma = np.mean(class_feat), np.std(class_feat)
                        fig, ax = plt.subplots(figsize=(7, 3))

                        # Istogramma
                        count, bins, _ = ax.hist(class_feat, bins=20, alpha=0.5, color=color, label=f"{name} ({len(class_feat)})", density=True)

                        # Gaussiana
                        x = np.linspace(bins[0], bins[-1], 200)
                        ax.plot(x, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2)), color=color, linewidth=2, label=f"Gaussiana {name}")

                        # Outlier (z-score > soglia)
                        z_scores = np.abs((class_feat - mu) / sigma)
                        out_feat = class_feat[z_scores > z_thr]
                        if len(out_feat) > 0:
                            ax.scatter(out_feat, [0.05]*len(out_feat), color='red', marker='x', s=100, label="Outlier")

                        # Soglie visive ±z_thr
                        ax.axvline(mu+z_thr*sigma, color=color, linestyle="--")
                        ax.axvline(mu-z_thr*sigma, color=color, linestyle="--")

                        ax.set_title(f"{name} – Distribution {feat_name}\n(outlier threshold ±{z_thr}σ)")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                
            else:
                st.info("Not enough labeled data yet to compute statistical outliers.")
                exclude_outlier = False
                outlier_idx = []

            # --- FEATURE VARIANCE ANALYSIS (after table) ---
            if st.session_state.show_var_analisi:
                st.subheader("📊 Statistical Feature Analysis by Class")
                st.info("""
                    Table summarizing key statistics (mean, standard deviation, variance) calculated for each feature, separately by class.
                    It provides a quick overview of how different features behave within classes, highlighting features with low variability or large differences between groups.
                    """)

                feedback_features = st.session_state.get("feedback_features", [])
                all_feature_names = st.session_state.get(
                    "all_feature_names",
                    [f"F{i+1}" for i in range(len(feedback_features[0]))]
                )
                df_var, outlier_df, df = analisi_varianza_feature_classificati(
                    feedback_features,
                    y_arr,
                    file_names,
                    all_feature_names,
                    zscore_thr=st.session_state.var_outlier_zscore
                )
                if df_var is not None:
                    st.write("Statistics (mean, std, var) for each feature and class:")
                    st.dataframe(df_var)
                else:
                    st.success("No statistical outliers detected in features (based on the current z-score threshold).")

            st.subheader("🔬 ANOVA Analysis Between Classes for Each Feature")
            st.info("""
                For each feature, an ANOVA (Analysis of Variance) test is performed to check if there is a significant difference
                in feature means between classes. The p-value column indicates significance: small values (e.g., <0.05)
                suggest that the feature effectively separates classes. Features with significant differences are marked with a check.
                """)

            # Prendi i dati già usati per analisi varianza
            #X = np.array(feature_list), y = np.array(label_list), feature_names = [f"F1",...]
            # Use the data already used for variance analysis
            # X = np.array(feature_list), y = np.array(label_list), feature_names = [f"F1", ...]
            if 'feedback_features' in st.session_state and len(st.session_state.feedback_features) > 0:
                X = np.array(st.session_state.feedback_features)
                y = np.array(st.session_state.feedback_labels)
                feature_names = st.session_state.get(
                    "all_feature_names",
                    [f"F{i+1}" for i in range(X.shape[1])]
                )

                # Ensure there are at least two classes and matching data
                if len(np.unique(y)) > 1 and X.shape[0] == len(y):
                    anova_results = []
                    for i, feat in enumerate(feature_names):
                        vals_0 = X[y == 0, i]
                        vals_1 = X[y == 1, i]
                        # Only if there is at least 1 point per class (otherwise scipy crashes)
                        if len(vals_0) > 1 and len(vals_1) > 1:
                            F, p = f_oneway(vals_0, vals_1)
                            anova_results.append({
                                "feature": feat,
                                "F_value": F,
                                "p_value": p,
                                "significant": "✅" if p < 0.05 else ""
                            })
                    if anova_results:
                        anova_df = pd.DataFrame(anova_results)
                        st.dataframe(anova_df)
                        st.info("Features with p-value < 0.05 (✅) have a significant difference between classes.")
                    else:
                        st.info("Not enough data for ANOVA test (need at least 2 points per class for each feature).")
                else:
                    st.info("ANOVA requires at least 2 classes with at least 2 points each.")
            else:
                st.info("Upload and label data to perform ANOVA analysis.")

            # ---- PLOT_BOXPLOT_FEATURE ----
            if "anova_df" in locals() and "df" in locals():
                # Select only features with p-value < 0.05
                sig_feat = anova_df.query("p_value < 0.05")["feature"].tolist()
                if len(sig_feat) == 0:
                    st.info("No feature shows a significant class difference according to ANOVA.")
                else:
                    st.subheader("🎯 Boxplot & Histograms for ANOVA-Significant Features")
                    st.info("""
                        For each feature significant in the ANOVA test (p-value < 0.05), show a boxplot 
                        of its distribution in the two classes and a comparison histogram.
                        This lets you compare shape, variability, and overlap between the two distributions.
                        """)

                    # -------- DROPDOWN MENU ---------
                    selected_feat = st.selectbox("Choose a feature", sig_feat)
                    if selected_feat:
                        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

                        # --- BOX PLOT
                        sns.boxplot(
                            data=df,
                            x="label",
                            y=selected_feat,
                            ax=axs[0],
                            palette=["#E57373", "#64B5F6"]
                        )
                        axs[0].set_xticklabels(["Different (0)", "Equal (1)"])
                        axs[0].set_title("Boxplot by Class")

                        # --- HISTOGRAM
                        for lbl, color, name in zip(
                            [0, 1],
                            ["#E57373", "#64B5F6"],
                            ["Different", "Equal"]
                        ):
                            sns.histplot(
                                df[df["label"] == lbl][selected_feat],
                                bins=15,
                                kde=True,
                                ax=axs[1],
                                color=color,
                                label=name,
                                stat="density",
                                alpha=0.6
                            )
                        axs[1].set_title("Histogram by Class")
                        axs[1].legend()
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                st.info("Both the ANOVA table and the full DataFrame with features are required.")

            if len(sig_feat) > 0:
                st.subheader(f"🎯 Main Effect + Variance (target plot) for: {selected_feat}")
                st.info("""
                    For the selected feature, display for each class the mean (as a point) and the variance 
                    (as an ‘aureola’ around the point). You can immediately see how much the class distributions 
                    overlap and how “wide” they are. If the aureolas are well separated, the ANOVA p-value will 
                    be low (feature is discriminative); if they overlap greatly, the p-value will be high. 
                    This plot helps visually interpret the ANOVA results.
                    """)


                main_effects_with_var_plot(df, [selected_feat])
            else:
                st.info("No feature is significant for the main effect.")

        def normalize_series(series):
                if series.nunique() <= 1 or series.isnull().all():
                    return series
                scaler = MinMaxScaler()
                normalized_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
                return pd.Series(normalized_data, index=series.index)

            # --- Funzione: Calcola i coefficienti di Fourier (assicurati che sia definita prima) ---
        def get_fourier_coefficients(y_data, num_harmonics=5):
                """
                Calcola i primi num_harmonics coefficienti di Fourier (ampiezze e fasi) per una serie di dati.
                Restituisce un array NumPy. Gestisce dati insufficienti.
                """
                # Rimuovi i valori NaN dalla serie y_data
                y_data_clean = y_data[~np.isnan(y_data)]
                
                if len(y_data_clean) < 2:
                    # Se i dati sono insufficienti per FFT, restituisce un array di zeri della dimensione attesa
                    return np.zeros(num_harmonics * 2) 
                
                fft_vals = np.fft.fft(y_data_clean)
                
                # Prendi solo i primi `num_harmonics` coefficienti (escludendo il componente DC e le frequenze negative)
                # Assicurati di non andare oltre la lunghezza reale dei dati FFT se num_harmonics è grande
                num_to_take = min(num_harmonics, len(fft_vals) // 2) # Considera solo la prima metà (positiva)

                amplitudes = np.abs(fft_vals[1 : num_to_take + 1])
                phases = np.angle(fft_vals[1 : num_to_take + 1])
                
                # Inizializza un array di zeri della dimensione finale desiderata (ampiezze + fasi)
                features = np.zeros(num_harmonics * 2) 
                
                # Popola l'array con le ampiezze e fasi calcolate
                features[:num_to_take] = amplitudes
                features[num_harmonics : num_harmonics + num_to_take] = phases
                
                return features


        # --- Funzione per estrarre feature da una curva (DataFrame) ---
        def extract_features(df, fourier_harmonics=5):
            features = {}
            for col in df.columns:
                # Solo colonne che sono curve (non scalari) e con dati sufficienti (almeno 2 punti per FFT)
                if df[col].nunique() > 1 and not df[col].isnull().all() and len(df[col].dropna()) >= 2: 
                    # Feature di base
                    features[f'{col}_mean'] = df[col].mean()
                    features[f'{col}_std'] = df[col].std()
                    features[f'{col}_min'] = df[col].min()
                    features[f'{col}_max'] = df[col].max()
                    
                    # Aggiungi coefficienti di Fourier
                    fourier_feats = get_fourier_coefficients(df[col].dropna().values, num_harmonics=fourier_harmonics)
                    for i, val in enumerate(fourier_feats):
                        if i < fourier_harmonics: # Amplitudes
                            features[f'{col}_fourier_amp_{i+1}'] = val
                        else: # Phases
                            features[f'{col}_fourier_phase_{i-fourier_harmonics+1}'] = val
            return features    
        def display_similarity_metrics(
        aligned_ref_features_series: pd.Series,
        aligned_test_features_series: pd.Series
        ):
            """
            Calcola e visualizza vari indici di somiglianza/distanza tra due serie di feature.

            Args:
                aligned_ref_features_series (pd.Series): Serie di feature della curva di riferimento, allineata.
                aligned_test_features_series (pd.Series): Serie di feature della curva di test, allineata.
            
            Returns:
                None: I risultati vengono visualizzati direttamente in Streamlit.
            """
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            st.subheader("📊 Similarity Metrics Between Reference Curve and Test Curve")
            with col1:
                # 1. Mean Absolute Difference
                overall_diff = np.mean(np.abs(aligned_ref_features_series - aligned_test_features_series))
                st.metric("1. **Mean Absolute Difference**", f"{overall_diff:.4f}", help="Mean of the absolute differences between feature values.")
            with col2:
                # 2. Euclidean Distance
                euclidean_dist = euclidean(aligned_ref_features_series, aligned_test_features_series)
                st.metric("2. **Euclidean Distance**", f"{euclidean_dist:.4f}", help="Straight-line distance between the feature vectors.")
            with col3:
                # 3. Cosine Similarity
                ref_norm = np.linalg.norm(aligned_ref_features_series)
                test_norm = np.linalg.norm(aligned_test_features_series)
                
                if ref_norm > 1e-9 and test_norm > 1e-9:  # Use a small threshold for floating-point values
                    cosine_sim = 1 - cosine(aligned_ref_features_series, aligned_test_features_series)
                    st.metric("3. **Cosine Similarity**", f"{cosine_sim:.4f}", help="Measures the angle between vectors (1 = identical, 0 = orthogonal, -1 = opposite).")
                else:
                    st.info("3. Cosine Similarity: Not computable (one or both feature vectors are near zero).")
            with col4:
                # 4. Manhattan Distance
                manhattan_dist = cityblock(aligned_ref_features_series, aligned_test_features_series)
                st.session_state.manhattan_dist = manhattan_dist
                st.metric("4. **Manhattan Distance**", f"{manhattan_dist:.4f}", help="Sum of absolute differences between feature values.")


            
        def train_curve_classifier(folder_path):   
            # 1) Load all CSVs into csv_data as (file_name, DataFrame)
            csv_data = []

            if isinstance(folder_path, (str, os.PathLike)):
                # Disk mode
                path_str = str(folder_path)
                if not os.path.isdir(path_str):
                    st.warning(f"No folder found: {path_str}")
                    return
                for fname in os.listdir(path_str):
                    if fname.lower().endswith('.csv'):
                        fpath = os.path.join(path_str, fname)
                        try:
                            df = pd.read_csv(fpath)
                            csv_data.append((fname, df))
                        except Exception as e:
                            st.error(f"Error reading {fname}: {e}")
                if not csv_data:
                    st.warning(f"No CSV files in: {path_str}")
                    return

            elif isinstance(folder_path, list):
                # Browser mode (UploadedFile list)
                for ufile in folder_path:
                    name = ufile.name
                    try:
                        text = ufile.getvalue().decode('utf-8')
                        df = pd.read_csv(io.StringIO(text))
                        csv_data.append((name, df))
                    except Exception as e:
                        st.error(f"Error uploading {name}: {e}")
                if not csv_data:
                    st.warning("No valid CSV uploaded via browser.")
                    return

            else:
                st.error("Invalid parameter: expected a path or list of UploadedFile.")
                return

            # --- From here on, use csv_data [(name, df), …] exactly as before ---
            # Example: reset rename counter
            if 'equal_csv_rename_counter' not in st.session_state:
                st.session_state.equal_csv_rename_counter = 0

            # Loading and normalization
            if not st.session_state.processed_dataframes:
                progress = st.progress(0, text="Loading and normalizing...")
                for i, (file_name, df) in enumerate(csv_data):
                    try:
                        for col in df.columns:
                            if df[col].nunique() > 1 and not df[col].isnull().all():
                                df[col] = normalize_series(df[col])
                        st.session_state.processed_dataframes[file_name] = df
                        progress.progress((i + 1) / len(csv_data), text=f"Normalized: {file_name}")
                    except Exception as e:
                        st.error(f"Error normalizing {file_name}: {e}")
                progress.empty()
                st.success(f"Loaded and normalized {len(st.session_state.processed_dataframes)} DataFrames.")

            # --- Populate st.session_state.all_feature_names ---
            # Collect all possible feature names from processed DataFrames
            temp_all_feature_names = set()
            if st.session_state.processed_dataframes:
                for df_data in st.session_state.processed_dataframes.values():
                    discovered_features = extract_features(df_data, fourier_harmonics=fourier_harmonics_slider)
                    temp_all_feature_names.update(discovered_features.keys())
                st.session_state.all_feature_names = sorted(list(temp_all_feature_names))
            else:
                st.session_state.all_feature_names = []  # No DataFrame, no features

            # --- Select Reference Curve ---
            if st.session_state.reference_curve_name is None and st.session_state.processed_dataframes:
                st.session_state.reference_curve_name = list(st.session_state.processed_dataframes.keys())[0]
                st.info(f"Reference curve set to: **{st.session_state.reference_curve_name}**")

            if not st.session_state.reference_curve_name:
                st.warning("No reference curve available for training.")
                return

            if st.session_state.processed_dataframes:
                reference_curve_name = st.selectbox(
                    "Select the reference curve:",
                    options=list(st.session_state.processed_dataframes.keys()),
                    index=0 if st.session_state.reference_curve_name is None else
                        list(st.session_state.processed_dataframes.keys()).index(st.session_state.reference_curve_name)
                )
                st.session_state.reference_curve_name = reference_curve_name

                reference_df = st.session_state.processed_dataframes[reference_curve_name]
                ref_features_dict = extract_features(reference_df, fourier_harmonics=fourier_harmonics_slider)

                with st.expander(f"Reference DataFrame: {reference_curve_name}"):
                    st.dataframe(reference_df.head())

                test_curve_names = [name for name in st.session_state.processed_dataframes.keys() if name != reference_curve_name]
            else:
                st.warning("No curves loaded.")
                st.stop()

            if not test_curve_names:
                st.info("No other curves to compare to the reference.")
                if not st.session_state.model_features_data.empty:
                    st.session_state.training_feedback_complete = True
                return

            if st.session_state.current_test_curve_index >= len(test_curve_names):
                st.info("You have compared all available curves.")
                st.session_state.current_test_curve_index = 0
                st.session_state.training_feedback_complete = True

            if not st.session_state.training_feedback_complete:
                current_test_name = test_curve_names[st.session_state.current_test_curve_index]
                current_test_df = st.session_state.processed_dataframes[current_test_name]
                current_test_features_dict = extract_features(current_test_df, fourier_harmonics=fourier_harmonics_slider)

                with st.expander(f"Current DataFrame: **{current_test_name}** ({st.session_state.current_test_curve_index + 1} of {len(test_curve_names)})"):
                    st.dataframe(current_test_df.head())

                # Compute a “distance” between features for manual feedback
                aligned_ref_features_series = pd.Series(ref_features_dict).reindex(st.session_state.all_feature_names, fill_value=0)
                aligned_test_features_series = pd.Series(current_test_features_dict).reindex(st.session_state.all_feature_names, fill_value=0)
                display_similarity_metrics(aligned_ref_features_series, aligned_test_features_series)
                overall_diff = np.mean(np.abs(aligned_ref_features_series - aligned_test_features_series))
                st.metric("Feature Mean Difference (for feedback)", f"{overall_diff:.4f}")

                with st.expander("Manual Training Info"):
                    st.info(
                        "Compare the two curves and decide if they are Equal using the buttons.\n"
                        "Note: If you mark them Equal, the software renames them with the same base name plus a sequence number, "
                        "and records the similarity value between the two curves."
                    )

                col_red, col_green = st.columns(2)
                with col_red:
                    if st.button("🔴 Different (NOT Equal)", use_container_width=True):
                        st.session_state.model_features_data.loc[len(st.session_state.model_features_data)] = [current_test_features_dict, 0]
                        st.session_state.current_test_curve_index += 1
                        st.info(f"File '{current_test_name}' kept with its original name.")
                        st.session_state.feedback_file_names.append(current_test_name)
                        st.session_state.feedback_labels.append(0)
                        st.session_state.feedback_features.append(aligned_test_features_series.values)
                        st.rerun()

                with col_green:
                    if st.button("🟢 Equal", use_container_width=True):
                        st.session_state.model_features_data.loc[len(st.session_state.model_features_data)] = [current_test_features_dict, 1]
                        st.session_state.current_test_curve_index += 1

                        # --- In-memory rename logic for Equal ---
                        base_name, ext = os.path.splitext(current_test_name)
                        ref_base_name, _ = os.path.splitext(st.session_state.reference_curve_name)

                        st.session_state.equal_csv_rename_counter += 1
                        new_file_name = f"{ref_base_name}_{st.session_state.equal_csv_rename_counter}_SimRef.csv"

                        if current_test_name in st.session_state.processed_dataframes:
                            df_to_update = st.session_state.processed_dataframes.pop(current_test_name)
                            st.session_state.processed_dataframes[new_file_name] = df_to_update

                        if "uploaded_csvs" in st.session_state:
                            new_uploaded_csvs = []
                            for f in st.session_state.uploaded_csvs:
                                if f.name == current_test_name:
                                    f.name = new_file_name
                                new_uploaded_csvs.append(f)
                            st.session_state.uploaded_csvs = new_uploaded_csvs

                        if st.session_state.reference_curve_name == current_test_name:
                            st.session_state.reference_curve_name = new_file_name

                        st.success(f"File '{current_test_name}' renamed to '{new_file_name}' in memory.")
                        st.session_state.feedback_file_names.append(current_test_name)
                        st.session_state.feedback_labels.append(1)
                        st.session_state.feedback_features.append(aligned_test_features_series.values)
                        st.rerun()

                # Show normalized curve comparison
                st.write("Normalized Curves Comparison:")
                fig, ax = plt.subplots(figsize=(10, 5))
                for col in reference_df.columns:
                    if reference_df[col].nunique() > 1 and not reference_df[col].isnull().all():
                        ax.plot(reference_df.index, reference_df[col], label=f'Reference - {col}')
                for col in current_test_df.columns:
                    if current_test_df[col].nunique() > 1 and not current_test_df[col].isnull().all():
                        ax.plot(current_test_df.index, current_test_df[col], linestyle='--', label=f'Test - {col}')
                ax.set_title("Normalized Curves Comparison")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.success("All curves have been evaluated! Proceed to train the model.")

            if st.session_state.model_features_data.empty:
                st.info("Label at least one curve to start training the model.")
                return

            # Add reference curve features (label 1) if not already added
            if 'ref_features_added' not in st.session_state or not st.session_state.ref_features_added:
                st.session_state.model_features_data.loc[len(st.session_state.model_features_data)] = [ref_features_dict, 1]
                st.session_state.ref_features_added = True

            # --- Align features before np.array(X) ---
            aligned_X_data = []
            if not st.session_state.all_feature_names:
                st.error("Feature names not available for alignment. Make sure they have been extracted.")
                return

            for feature_dict in st.session_state.model_features_data['features']:
                aligned_series = pd.Series(feature_dict).reindex(st.session_state.all_feature_names, fill_value=0)
                aligned_X_data.append(aligned_series.values)

            X = np.array(aligned_X_data)
            y = st.session_state.model_features_data['label'].values.astype(int)

            # Ensure at least two classes and enough samples
            if len(np.unique(y)) < 2:
                st.warning("Need feedback for at least two classes (Equal and Different) to train a classifier.")
                return
            if len(X) < 2:
                st.warning("Need at least two samples to train the model.")
                return

            if st.button("Train Model", type="primary"):
                try:
                    st.info(f"Training model: {model_choice}...")

                    if model_choice == "Support Vector Machine (SVC)":
                        model = SVC(random_state=42, **st.session_state.model_params)
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(random_state=42, **st.session_state.model_params)
                    elif model_choice == "Random Forest":
                        model = RandomForestClassifier(random_state=42, **st.session_state.model_params)
                    elif model_choice == "Neural Network (MLP)":
                        model = MLPClassifier(random_state=42, max_iter=1000, **st.session_state.model_params)

                    # Optional train/test split for realistic evaluation
                    if len(X) > 1 and len(np.unique(y)) > 1:
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                        except ValueError:
                            st.warning("Insufficient data for train/test split. Using all data for training.")
                            X_train, y_train = X, y
                            X_test, y_test = X, y
                    else:
                        X_train, y_train = X, y
                        X_test, y_test = X, y

                    model.fit(X_train, y_train)
                    st.session_state.trained_model = model

                    # Model evaluation
                    y_pred = model.predict(X_test)
                    report = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, zero_division=0),
                        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
                    }
                    st.session_state.model_report = report

                    st.success("Model trained successfully!")
                    st.json(report)

                    # Show feature importances for Random Forest
                    if model_choice == "Random Forest" and hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importances (Random Forest)")
                        feature_names = st.session_state.all_feature_names
                        if len(feature_names) == len(model.feature_importances_):
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': model.feature_importances_
                            }).sort_values(by='Importance', ascending=False)
                            st.dataframe(importance_df)
                            fig_imp, ax_imp = plt.subplots()
                            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax_imp)
                            ax_imp.set_title("Top 10 Feature Importances")
                            st.pyplot(fig_imp)
                            plt.close(fig_imp)
                        else:
                            st.warning("Cannot display feature importances: feature count mismatch.")

                except Exception as e:
                    st.error(f"Error training model: {e}")
                    st.session_state.trained_model = None
                    st.session_state.model_report = None

            # Section runs only if a model has been trained
            if st.session_state.trained_model is not None:
                st.subheader("Trained Model and Report")
                if st.session_state.model_report:
                    st.json(st.session_state.model_report)

                # --- Export Model and Parameters ---
                st.subheader("Export Model and Parameters")

                model_filename = f"{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}_trained_model.joblib"
                joblib_buffer = pickle.dumps(st.session_state.trained_model)
                st.download_button(
                    label="Download Trained Model (.joblib)",
                    data=joblib_buffer,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )

                report_json_filename = f"{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}_report.json"
                st.download_button(
                    label="Download Model Report (JSON)",
                    data=json.dumps(st.session_state.model_report, indent=2),
                    file_name=report_json_filename,
                    mime="application/json"
                )

                training_params_json_filename = f"{model_choice.replace(' ', '_').replace('(', '').replace(')', '')}_params.json"
                export_params = {
                    "model_choice": model_choice,
                    "model_parameters": st.session_state.model_params,
                    "fourier_harmonics": fourier_harmonics_slider,
                    "feature_names": st.session_state.all_feature_names
                }
                st.download_button(
                    label="Download Training Parameters (JSON)",
                    data=json.dumps(export_params, indent=2),
                    file_name=training_params_json_filename,
                    mime="application/json"
                )

            else:
                st.info("Train the model to enable export.")

            # Reset training and feedback data
            if st.button("Reset Training and Feedback Data"):
                st.session_state.processed_dataframes = {}
                st.session_state.reference_curve_name = None
                st.session_state.current_test_curve_index = 0
                st.session_state.training_feedback = []
                st.session_state.model_features_data = pd.DataFrame(columns=['features', 'label'])
                st.session_state.trained_model = None
                st.session_state.model_report = None
                if 'ref_features_added' in st.session_state:
                    del st.session_state.ref_features_added
                st.info("Training and feedback reset. Reload the page to start over.")
                st.rerun()
        
            
            # Inizializza il path solo una volta
        if "folder_path" not in st.session_state:
            st.session_state.folder_path = "C:/Users/mdirienzo/Desktop/Mirco/addestramento cane"

    with st.expander("ℹ️ Info"):
        st.write(
            "Module B: Train a model from CSV dataframes by choosing among various training methods in the *SIDEBAR Module B* menus and adjusting parameters with sliders.\n"
            "- Load all CSV dataframes at once via multi-select in the Browser (created in *ModuleA* or external).\n"
            "- Each file corresponds to one sample (one image), is normalized and compared to the others; the first is the reference model.\n"
            "- Each time the user labels a pair as equal or different, the model is retrained.\n"
            "- By fitting scatter plots of various methods you can see in 2D which training model suits the data best.\n"
            "- Export the trained model."
        )

    with st.expander("📁 Input CSV File Selection"):
        st.info("Upload the CSV files you want to use for training")
        uploaded_files = st.file_uploader(
            label="Select one or more CSV files",
            type="csv",
            accept_multiple_files=True
        )
        # --- AFTER FILE UPLOADER, save an original copy ---
        if uploaded_files:
            if "original_dataframes" not in st.session_state or not st.session_state.original_dataframes:
                st.session_state.original_dataframes = {}
                for ufile in uploaded_files:
                    df = pd.read_csv(ufile)
                    st.session_state.original_dataframes[ufile.name] = df.copy()
        if uploaded_files:
            # Store the UploadedFile objects
            st.session_state.uploaded_csvs = uploaded_files
            # Also store their names separately
            st.session_state.file_names_list = [f.name for f in uploaded_files]
            st.success(f"✅ Loaded {len(uploaded_files)} CSV files!")
        else:
            st.info("📌 No files uploaded")

    # Display the list of uploaded files under the expander
    if "uploaded_csvs" in st.session_state and st.session_state.uploaded_csvs:
        file_names = [f.name for f in st.session_state.uploaded_csvs]

    # --- 4. CALL THE FUNCTION HERE ---
    # Pass the list of uploaded files to the function
    if "uploaded_csvs" in st.session_state and st.session_state.uploaded_csvs:
        train_curve_classifier(st.session_state.uploaded_csvs)
        if not st.session_state.model_features_data.empty:
            aligned_X_data = []
            for feature_dict in st.session_state.model_features_data['features']:
                aligned_series = pd.Series(feature_dict).reindex(
                    st.session_state.all_feature_names, fill_value=0
                )
                aligned_X_data.append(aligned_series.values)
            X = np.array(aligned_X_data)
            y = st.session_state.model_features_data['label'].values.astype(int)

            model = get_model(model_choice, st.session_state.model_params)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # --- Compute statistical outliers and save to session_state ---
            z_thr = st.session_state.get("var_outlier_zscore", 2.5)
            outlier_idx = []
            for lbl in [0, 1]:
                mask = (y == lbl)
                if mask.sum() > 0:
                    class_feat = X[mask]
                    z_scores = np.abs(zscore(class_feat, axis=0, nan_policy='omit'))
                    max_z = np.nanmax(z_scores, axis=1)
                    idx_this_class = np.where(mask)[0]
                    class_outlier_idx = idx_this_class[max_z > z_thr]
                    outlier_idx.extend(class_outlier_idx.tolist())
            st.session_state['outlier_idx'] = outlier_idx  # <-- save here

            # --- Train model and plot decision boundary ---
            model.fit(X_pca, y)
            plot_decision_boundary(model, X_pca, y)
        else:
            st.info("Label some curves and train the model to view the decision boundary.")
    else:
        st.info("Upload at least one CSV file to start training.")

    with tab3:
        st.markdown(tab_labels[2], unsafe_allow_html=True)
        with st.expander("Upload Parameters CSV"):
            st.info(
                "Assuming you have created at least one DataFrame (DF) in *Module A* that includes the SIDEBAR parameters used for analysis, "
                "you can upload that sample DF here to set parameters and perform new analyses on images in batch."
            )
            file_csv = st.file_uploader("Upload a CSV file", type="csv", key="csv_Module_c")
            if file_csv is not None:
                if st.button("Apply Parameters (save for sidebar)"):
                    st.session_state.pending_params_file_bytes = file_csv.getvalue()
                    st.success("File ready—will be applied via the sidebar!")
                    st.rerun()

        st.divider()
        with st.expander("Upload Images"):
            st.info(
                "Upload multiple images to automatically analyze each one using the sidebar parameters "
                "set via the sample CSV you uploaded above.\n"
                "Download the resulting image analysis DataFrames as a zipped folder."
            )
            uploaded_images = st.file_uploader(
                "Upload images (PNG, JPG, JPEG) — multiple or zip",
                type=["png", "jpg", "jpeg", "zip"],
                accept_multiple_files=True,
                key="multi_img_upload"
            )
            uploaded_zip = st.file_uploader("or ZIP of images", type=["zip"], key="batch_zip")
            anteprime_imgs = []
            anteprime_names = []

            if st.button("Run Batch on All Images"):
                if not uploaded_images and not uploaded_zip:
                    st.error("Upload images (or a ZIP) before running the batch!")
                else:
                    # --- Always use the sidebar parameters ---
                    params = st.session_state.sidebar_params.copy()
                    st.info(
                        "🟢 Batch parameters taken from the current sidebar. "
                        "To change them, adjust the sidebar or upload a parameters CSV first."
                    )


                    # --- Colleziona immagini ---
                    images_to_process = []
                    if uploaded_images:
                        images_to_process.extend(uploaded_images)
                    if uploaded_zip:
                        with zipfile.ZipFile(uploaded_zip) as z:
                            for name in z.namelist():
                                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    buf = BytesIO(z.read(name))
                                    buf.name = name
                                    images_to_process.append(buf)                                
                    st.info(f"Found {len(images_to_process)} image to process.")

                    # --- Buffer ZIP risultati ---
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_out:
                        for img_stream in images_to_process:
                            try:
                                img_stream.seek(0)
                                file_bytes = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
                                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                                df_final_download, img_final = process_image_and_get_df(img_bgr, params, getattr(img_stream, "name", "img"))
                                image_base_name = img_stream.name.replace('.', '_')

                                # Salva le anteprime
                                anteprime_imgs.append(img_final)
                                anteprime_names.append(img_stream.name)

                                excel_buf = BytesIO()
                                with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
                                    df_excel = df_final_download.copy()
                                    for col in df_excel.columns:
                                        if col.startswith("Param_"):
                                            df_excel.loc[1:, col] = ""
                                    df_excel.to_excel(writer, sheet_name="Dati", index=False)

                                    params_df = pd.DataFrame(
                                        st.session_state.sidebar_params.items(),
                                        columns=["Parametro", "Valore"]
                                    )
                                    params_df.to_excel(writer, sheet_name="Parametri", index=False)
                                excel_buf.seek(0)

                                df_from_excel = pd.read_excel(excel_buf, sheet_name="Dati")
                                csv_buf = BytesIO()
                                df_from_excel.to_csv(csv_buf, index=False)
                                csv_buf.seek(0)

                                zip_out.writestr(f"{image_base_name}_risultato.csv", csv_buf.read())

                            except Exception as e:
                                st.warning(f"Image elaboration error {img_stream.name}: {e}")

                    st.success("Batch finish!")
                    zip_buffer.seek(0)
                    st.download_button(
                        "Download result into ZIP format",
                        data=zip_buffer.getvalue(),
                        file_name="batch_risultati.zip",
                        mime="application/zip"
                    )
                if anteprime_imgs:  # Se ci sono immagini elaborate
                    st.subheader("📸 Elaborated Images Preview")
                    ncols = 4
                    rows = (len(anteprime_imgs) + ncols - 1) // ncols
                    for i in range(rows):
                        cols = st.columns(ncols)
                        for j in range(ncols):
                            idx = i * ncols + j
                            if idx < len(anteprime_imgs):
                                with cols[j]:
                                    st.image(anteprime_imgs[idx], 
                                            caption=anteprime_names[idx], 
                                            use_container_width=True)
        st.divider()
        with st.expander("📂 Continue training existing model from CSV ZIP"):
            zip_csv_file = st.file_uploader(
                "Upload ZIP of CSV files",
                type="zip",
                key="zip_csv_tab3"
            )
            col1, col2 = st.columns(2)
            with col1:
                model_file = st.file_uploader(
                    "Trained model (.joblib/.pkl)",
                    type=["joblib", "pkl"],
                    key="load_model_tab3"
                )
            with col2:
                params_file = st.file_uploader(
                    "Model parameters (.json)",
                    type=["json"],
                    key="load_params_tab3"
                )

            snippet = st.text_input(
                "Text snippet to search in CSV filename → label = 1",
                help="Assigns label 1 if snippet is found, otherwise 0."
            )

            if zip_csv_file and model_file and params_file and snippet:
                try:
                    # 🔹 Load parameters file
                    params_file.seek(0)
                    raw_params = params_file.read()
                    st.write(f"First bytes of parameters file: {raw_params[:20]}")
                    params = json.load(io.BytesIO(raw_params))

                    feature_names = params.get("feature_names", [])
                    fourier_harm = params.get("fourier_harmonics", None)

                    # 🔹 Load model
                    raw_model = model_file.read()
                    try:
                        model = joblib.load(BytesIO(raw_model))
                    except Exception:
                        model = pickle.loads(raw_model)

                    # 🔹 Unzip and process CSV files
                    zip_data = zip_csv_file.read()
                    temp_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(BytesIO(zip_data)) as z:
                        z.extractall(temp_dir)

                    csv_streams = []
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(".csv"):
                                file_path = os.path.join(root, file)
                                try:
                                    df = pd.read_csv(file_path)
                                except UnicodeDecodeError:
                                    df = pd.read_csv(file_path, encoding='latin1')
                                except Exception as e:
                                    st.warning(f"⚠ Unable to read {file}: {e}")
                                    continue
                                csv_streams.append((file, df))

                    if not csv_streams:
                        st.error("❌ No valid CSV found in ZIP.")
                        st.stop()

                    X_list, y_list = [], []
                    for name, df in csv_streams:
                        # Normalize each numeric column
                        for col in df.columns:
                            if df[col].nunique() > 1 and not df[col].isnull().all():
                                df[col] = normalize_series(df[col])
                        feats = extract_features(df, fourier_harmonics=fourier_harm)
                        aligned = pd.Series(feats).reindex(feature_names, fill_value=0).values
                        X_list.append(aligned)
                        label = 1 if snippet.lower() in name.lower() else 0
                        y_list.append(label)

                    X = np.vstack(X_list)
                    y = np.array(y_list)

                    if len(np.unique(y)) < 2:
                        st.error("At least one CSV must be labeled 0 and one labeled 1.")
                        st.stop()

                    X_pca = PCA(n_components=2).fit_transform(X)
                    model.fit(X_pca, y)
                    st.success("✔️ Model updated successfully!")
                    st.session_state.feedback_file_names = [name for name, df in csv_streams]
                    st.session_state.feedback_labels = y.tolist()
                    st.session_state.feedback_features = X.tolist()
                    # Update feature names if provided
                    if feature_names:
                        st.session_state.all_feature_names = feature_names
                    else:
                        st.session_state.all_feature_names = [f"F{i+1}" for i in range(X.shape[1])]

                    # Recompute outliers
                    z_thr = st.session_state.get("var_outlier_zscore", 2.5)
                    outlier_idx = []
                    for lbl in [0, 1]:
                        mask = (y == lbl)
                        if mask.sum() > 0:
                            class_feat = X[mask]
                            z_scores = np.abs(zscore(class_feat, axis=0, nan_policy='omit'))
                            max_z = np.nanmax(z_scores, axis=1)
                            idxs = np.where(mask)[0]
                            outlier_idx.extend(idxs[max_z > z_thr].tolist())
                    st.session_state['outlier_idx'] = outlier_idx

                    plot_decision_boundary(model, X_pca, y)

                    # Offer updated model for download
                    out_buf = BytesIO()
                    pickle.dump(model, out_buf)
                    out_buf.seek(0)
                    st.download_button(
                        "Download updated model (.joblib)",
                        data=out_buf,
                        file_name="model_updated.joblib",
                        mime="application/octet-stream"
                    )

                except Exception as e:
                    st.error(f"Retraining error: {e}")
            else:
                st.info("Upload ZIP, model, JSON and enter snippet to start retraining.")

        st.divider()

        with st.expander("🤖 Classify batch results with a pre-trained model"):
            st.info(
                "Apply your already trained model (.joblib/.pkl + .json parameters) "
                "to the CSVs generated by the image batch."
            )

            col1, col2 = st.columns(2)
            with col1:
                pred_model_file = st.file_uploader(
                    "Trained model (.joblib/.pkl)", type=["joblib", "pkl"], key="pred_model_tab3"
                )
            with col2:
                pred_params_file = st.file_uploader(
                    "Model parameters (.json)", type=["json"], key="pred_params_tab3"
                )
            uploaded_csv_zip = st.file_uploader(
                "ZIP of batch result CSVs", type="zip", key="csv_pred_zip_tab3"
            )

            if not (pred_model_file and pred_params_file and uploaded_csv_zip):
                st.info("Upload model, parameters, and ZIP of CSVs to proceed.")
                st.stop()

            # Load model and parameters
            params = json.load(io.BytesIO(pred_params_file.read()))
            feature_names = params.get("feature_names", [])
            fourier_harm = params.get("fourier_harmonics")
            raw = pred_model_file.read()
            try:
                model = joblib.load(BytesIO(raw))
            except:
                model = pickle.loads(raw)

            # Extract features and predict
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded_csv_zip) as z:
                z.extractall(tmp)

            X_pred, file_pred_names = [], []
            for fname in os.listdir(tmp):
                if not fname.lower().endswith(".csv"):
                    continue
                try:
                    df = pd.read_csv(os.path.join(tmp, fname))
                    for c in df.columns:
                        if df[c].nunique() > 1 and not df[c].isnull().all():
                            df[c] = normalize_series(df[c])
                    feats = extract_features(df, fourier_harmonics=fourier_harm)
                    aligned = pd.Series(feats).reindex(feature_names, fill_value=0).values
                    X_pred.append(aligned)
                    file_pred_names.append(fname)
                except Exception as e_file:
                    st.warning(f"Unable to process {fname}: {e_file}")

            if not X_pred:
                st.error("No valid CSV found.")
                st.stop()

            X_pred = np.vstack(X_pred)
            exp_n = getattr(model, "n_features_in_", X_pred.shape[1])
            if X_pred.shape[1] != exp_n:
                X_final = PCA(n_components=exp_n).fit_transform(X_pred)
            else:
                X_final = X_pred

            pred_labels = model.predict(X_final)

            # Build initial output DataFrame
            df_out = pd.DataFrame({
                "File": file_pred_names,
                "Predicted": pred_labels
            })
            col1, col2 = st.columns([1, 1])
            with col1:
                wrong = st.multiselect(
                    "Select files with WRONG predictions",
                    options=file_pred_names
                )

                true_labels = {}
                if wrong:
                    st.write("**For each wrongly predicted file, select the TRUE class:**")
                    for f in wrong:
                        true_labels[f] = st.selectbox(
                            f,
                            options=model.classes_.tolist(),
                            key=f"true_{f}"
                        )

            # Compute TrueLabel column
            def get_true_label(row):
                return true_labels.get(row.File, row.Predicted)

            df_out["TrueLabel"] = df_out.apply(get_true_label, axis=1)

            # Highlight wrong rows in red
            def highlight_wrong(row):
                return ["color: red" if row.File in wrong else "" for _ in row]

            # Confusion matrix and report
            y_true = df_out["TrueLabel"]
            y_pred = df_out["Predicted"]

            report = classification_report(y_true, y_pred, output_dict=True)
            st.write("**Classification report:**")
            st.table(pd.DataFrame(report).T)

            class0, class1 = model.classes_.tolist()

            def invert_label(row):
                if row.File in wrong:
                    return class1 if row.Predicted == class0 else class0
                return row.Predicted

            df_out["TrueLabel"] = df_out.apply(invert_label, axis=1)

            with col2:
                st.write("**Predictions with feedback highlighted:**")
                st.table(df_out.style.apply(highlight_wrong, axis=1))

            # Confusion matrix table
            cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
            cm_df = pd.DataFrame(
                cm,
                index=[f"True {c}" for c in model.classes_],
                columns=[f"Pred {c}" for c in model.classes_]
            )
            st.write("**Confusion matrix:**")
            st.table(cm_df)

            # Plot decision boundary and offer CSV download
            plot_decision_boundary(model, X_final, pred_labels)

            buf = BytesIO()
            df_out.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button(
                "Download CSV with feedback and TrueLabel",
                data=buf,
                file_name="predictions_with_feedback.csv",
                mime="text/csv"
            )
