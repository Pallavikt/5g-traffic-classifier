"""
5G Network Traffic Classifier
- Synthetic Dataset Generation
- EDA (Exploratory Data Analysis) with Plots
- 4 ML Algorithms: Random Forest, SVM, KNN, XGBoost
- Tkinter GUI Frontend
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import warnings
warnings.filterwarnings('ignore')

# ── Data & ML ──────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
import joblib

# ── Plotting ───────────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# ══════════════════════════════════════════════════════════════
# 1.  DATASET GENERATION
# ══════════════════════════════════════════════════════════════

TRAFFIC_CLASSES = [
    "Video_Streaming",
    "VoIP",
    "Web_Browsing",
    "Online_Gaming",
    "IoT_Sensor",
    "File_Transfer",
    "Video_Conferencing",
]

def generate_5g_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Per-class statistical profiles  [mean, std]
    profiles = {
        "Video_Streaming":    dict(throughput=(85, 15),  latency=(20, 5),   jitter=(2, 0.5),
                                   packet_loss=(0.5, 0.2),  packet_size=(1400, 100), burst_ratio=(0.7, 0.1),
                                   flow_duration=(300, 60),  signal_strength=(-70, 5)),
        "VoIP":               dict(throughput=(0.1, 0.05), latency=(10, 3),  jitter=(1, 0.3),
                                   packet_loss=(0.2, 0.1),  packet_size=(160, 20),  burst_ratio=(0.3, 0.05),
                                   flow_duration=(120, 30),  signal_strength=(-75, 6)),
        "Web_Browsing":       dict(throughput=(5, 3),    latency=(50, 15),  jitter=(5, 2),
                                   packet_loss=(1, 0.5),   packet_size=(800, 200), burst_ratio=(0.4, 0.1),
                                   flow_duration=(30, 10),   signal_strength=(-72, 5)),
        "Online_Gaming":      dict(throughput=(3, 1),    latency=(15, 4),   jitter=(1.5, 0.4),
                                   packet_loss=(0.3, 0.1),  packet_size=(200, 50),  burst_ratio=(0.5, 0.1),
                                   flow_duration=(600, 120), signal_strength=(-68, 4)),
        "IoT_Sensor":         dict(throughput=(0.01,0.005),latency=(100, 30), jitter=(10, 3),
                                   packet_loss=(2, 1),     packet_size=(64, 16),   burst_ratio=(0.1, 0.02),
                                   flow_duration=(10, 3),    signal_strength=(-85, 8)),
        "File_Transfer":      dict(throughput=(50, 20),  latency=(30, 10),  jitter=(3, 1),
                                   packet_loss=(0.1, 0.05), packet_size=(1500, 50), burst_ratio=(0.9, 0.05),
                                   flow_duration=(60, 20),   signal_strength=(-71, 5)),
        "Video_Conferencing": dict(throughput=(4, 1),    latency=(25, 6),   jitter=(2, 0.5),
                                   packet_loss=(0.4, 0.15), packet_size=(900, 150), burst_ratio=(0.6, 0.08),
                                   flow_duration=(1800,300), signal_strength=(-69, 4)),
    }

    rows = []
    samples_per_class = n_samples // len(TRAFFIC_CLASSES)

    for label, p in profiles.items():
        n = samples_per_class
        def s(key):
            val = rng.normal(p[key][0], p[key][1], n)
            return np.clip(val, 0, None)

        rows.append(pd.DataFrame({
            "throughput_mbps":    s("throughput"),
            "latency_ms":         s("latency"),
            "jitter_ms":          s("jitter"),
            "packet_loss_pct":    s("packet_loss"),
            "avg_packet_size_B":  s("packet_size"),
            "burst_ratio":        np.clip(s("burst_ratio"), 0, 1),
            "flow_duration_s":    s("flow_duration"),
            "signal_strength_dBm":rng.normal(p["signal_strength"][0], p["signal_strength"][1], n),
            "packets_per_second": rng.exponential(100, n) * (p["throughput"][0] / 10 + 0.1),
            "retransmission_rate":np.clip(rng.normal(p["packet_loss"][0]*0.5, 0.1, n), 0, 1),
            "traffic_class":      label,
        }))

    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# 2.  MODEL TRAINING
# ══════════════════════════════════════════════════════════════

MODELS = {
    "Random Forest":   RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "SVM":             SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    "KNN":             KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
    "Gradient Boost":  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

FEATURE_COLS = [
    "throughput_mbps", "latency_ms", "jitter_ms", "packet_loss_pct",
    "avg_packet_size_B", "burst_ratio", "flow_duration_s",
    "signal_strength_dBm", "packets_per_second", "retransmission_rate",
]

class TrafficClassifierApp:
    # ── Colours & Fonts ───────────────────────────────────────
    BG       = "#0a0e1a"
    PANEL    = "#111827"
    ACCENT   = "#00d4ff"
    ACCENT2  = "#7c3aed"
    SUCCESS  = "#10b981"
    WARN     = "#f59e0b"
    TEXT     = "#e2e8f0"
    SUBTEXT  = "#94a3b8"
    FONT     = ("Consolas", 11)
    FONT_B   = ("Consolas", 11, "bold")
    TITLE_F  = ("Consolas", 22, "bold")

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("5G Traffic Classifier — ML Dashboard")
        self.root.configure(bg=self.BG)
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)

        self.df        = None
        self.results   = {}
        self.scaler    = StandardScaler()
        self.le        = LabelEncoder()
        self.trained   = {}

        self._build_ui()

    # ─── UI Construction ──────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────────────
        hdr = tk.Frame(self.root, bg=self.BG, pady=12)
        hdr.pack(fill="x", padx=20)

        tk.Label(hdr, text="⬡  5G TRAFFIC CLASSIFIER", font=self.TITLE_F,
                 fg=self.ACCENT, bg=self.BG).pack(side="left")
        tk.Label(hdr, text="ML-Powered Network Intelligence",
                 font=("Consolas", 12), fg=self.SUBTEXT, bg=self.BG).pack(side="left", padx=20)

        # status dot
        self.status_var = tk.StringVar(value="● IDLE")
        self.status_lbl = tk.Label(hdr, textvariable=self.status_var, font=self.FONT_B,
                 fg=self.WARN, bg=self.BG)
        self.status_lbl.pack(side="right")

        # ── Notebook ─────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",          background=self.BG,  borderwidth=0)
        style.configure("TNotebook.Tab",      background=self.PANEL, foreground=self.SUBTEXT,
                        font=self.FONT_B, padding=[18, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", self.ACCENT2)],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self.tab_ctrl   = self._make_frame(nb);  nb.add(self.tab_ctrl,  text="  ⚙  Control  ")
        self.tab_eda    = self._make_frame(nb);  nb.add(self.tab_eda,   text="  📊  EDA  ")
        self.tab_train  = self._make_frame(nb);  nb.add(self.tab_train, text="  🤖  Training  ")
        self.tab_pred   = self._make_frame(nb);  nb.add(self.tab_pred,  text="  🔍  Predict  ")
        self.tab_perf   = self._make_frame(nb);  nb.add(self.tab_perf,  text="  📈  Performance  ")

        self._build_control_tab()
        self._build_eda_tab()
        self._build_train_tab()
        self._build_pred_tab()
        self._build_perf_tab()

    def _make_frame(self, parent):
        f = tk.Frame(parent, bg=self.BG)
        return f

    def _card(self, parent, title="", row=0, col=0, rowspan=1, colspan=1, sticky="nsew"):
        outer = tk.Frame(parent, bg=self.PANEL, bd=0,
                         highlightbackground=self.ACCENT2, highlightthickness=1)
        outer.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan,
                   padx=8, pady=8, sticky=sticky)
        if title:
            tk.Label(outer, text=title, font=self.FONT_B, fg=self.ACCENT,
                     bg=self.PANEL).pack(anchor="w", padx=12, pady=(10, 4))
        inner = tk.Frame(outer, bg=self.PANEL)
        inner.pack(fill="both", expand=True, padx=10, pady=6)
        return inner

    def _btn(self, parent, text, cmd, color=None, **kw):
        c = color or self.ACCENT2
        b = tk.Button(parent, text=text, command=cmd,
                      font=self.FONT_B, fg="white", bg=c,
                      activebackground=self.ACCENT, activeforeground=self.BG,
                      relief="flat", cursor="hand2", padx=12, pady=6, **kw)
        return b

    # ─── TAB 1 : CONTROL ──────────────────────────────────────
    def _build_control_tab(self):
        t = self.tab_ctrl
        t.columnconfigure((0, 1, 2), weight=1)
        t.rowconfigure(2, weight=1)

        # Dataset generation card
        c = self._card(t, "📦  Generate Dataset", row=0, col=0)
        tk.Label(c, text="Samples:", font=self.FONT, fg=self.TEXT, bg=self.PANEL).grid(row=0, column=0, sticky="w")
        self.n_samples_var = tk.IntVar(value=5000)
        ttk.Spinbox(c, from_=1000, to=20000, increment=500,
                    textvariable=self.n_samples_var, width=10,
                    font=self.FONT).grid(row=0, column=1, padx=8, pady=4)
        self._btn(c, "⚡  Generate Dataset", self._generate_dataset).grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)

        # Load dataset card
        c_load = self._card(t, "📂  Load Dataset (CSV)", row=1, col=0, colspan=3)
        # file path display
        self.load_path_var = tk.StringVar(value="No file selected")
        path_lbl = tk.Label(c_load, textvariable=self.load_path_var, font=("Consolas", 9),
                            fg=self.SUBTEXT, bg=self.PANEL, anchor="w")
        path_lbl.grid(row=0, column=0, columnspan=3, sticky="ew", padx=4, pady=(2, 4))

        self._btn(c_load, "📁  Browse CSV File", self._browse_csv, color="#0f766e").grid(
            row=1, column=0, sticky="ew", padx=4, pady=4)

        # label column selector
        tk.Label(c_load, text="Label column:", font=self.FONT, fg=self.TEXT,
                 bg=self.PANEL).grid(row=1, column=1, sticky="e", padx=(16, 4))
        self.label_col_var = tk.StringVar(value="traffic_class")
        self.label_col_entry = tk.Entry(c_load, textvariable=self.label_col_var,
                                        font=self.FONT, bg="#1e293b", fg=self.TEXT,
                                        insertbackground=self.ACCENT, relief="flat", width=18)
        self.label_col_entry.grid(row=1, column=2, sticky="w", padx=(0, 8), pady=4)

        self._btn(c_load, "✅  Load & Validate", self._load_csv, color="#1d4ed8").grid(
            row=2, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        # dataset info strip
        self.ds_info_var = tk.StringVar(value="")
        tk.Label(c_load, textvariable=self.ds_info_var, font=("Consolas", 9),
                 fg=self.SUCCESS, bg=self.PANEL, anchor="w").grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 4))

        c_load.columnconfigure((0, 1, 2), weight=1)

        # Training card
        c2 = self._card(t, "🤖  Model Training", row=0, col=1)
        tk.Label(c2, text="Test Size:", font=self.FONT, fg=self.TEXT, bg=self.PANEL).grid(row=0, column=0, sticky="w")
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(c2, from_=0.1, to=0.4, increment=0.05,
                    textvariable=self.test_size_var, width=10,
                    font=self.FONT).grid(row=0, column=1, padx=8, pady=4)
        self._btn(c2, "🚀  Train All Models", self._train_models).grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)

        # Quick actions
        c3 = self._card(t, "🎯  Quick Actions", row=0, col=2)
        self._btn(c3, "📊  Run EDA",         self._run_eda,     color="#0f766e").pack(fill="x", pady=3)
        self._btn(c3, "📈  Plot Performance", self._plot_perf,  color="#7c3aed").pack(fill="x", pady=3)
        self._btn(c3, "💾  Save Models",      self._save_models, color="#1d4ed8").pack(fill="x", pady=3)

        # Log
        log_frame = self._card(t, "📋  Activity Log", row=2, col=0, colspan=3)
        self.log = scrolledtext.ScrolledText(
            log_frame, font=("Consolas", 10), bg="#0d1117", fg="#c9d1d9",
            insertbackground=self.ACCENT, relief="flat", state="disabled"
        )
        self.log.pack(fill="both", expand=True)
        self._log("System ready.  Generate a dataset OR load a CSV to begin.")

    # ─── TAB 2 : EDA ──────────────────────────────────────────
    def _build_eda_tab(self):
        t = self.tab_eda
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)
        tk.Label(t, text="Generate dataset and click 'Run EDA' to see plots.",
                 font=self.FONT, fg=self.SUBTEXT, bg=self.BG).pack(pady=40)

    # ─── TAB 3 : TRAINING ─────────────────────────────────────
    def _build_train_tab(self):
        t = self.tab_train
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)
        tk.Label(t, text="Train models first to see results here.",
                 font=self.FONT, fg=self.SUBTEXT, bg=self.BG).pack(pady=40)

    # ─── TAB 4 : PREDICT ──────────────────────────────────────
    def _build_pred_tab(self):
        t = self.tab_pred
        t.columnconfigure((0, 1), weight=1)

        c = self._card(t, "🔢  Enter Network Metrics", row=0, col=0, sticky="nsew")
        self.pred_vars = {}
        defaults = {
            "throughput_mbps": 50.0, "latency_ms": 20.0, "jitter_ms": 2.0,
            "packet_loss_pct": 0.5,  "avg_packet_size_B": 1200.0, "burst_ratio": 0.7,
            "flow_duration_s": 300.0,"signal_strength_dBm": -70.0,
            "packets_per_second": 500.0, "retransmission_rate": 0.3,
        }
        for i, (feat, val) in enumerate(defaults.items()):
            tk.Label(c, text=feat, font=self.FONT, fg=self.TEXT, bg=self.PANEL).grid(
                row=i, column=0, sticky="w", pady=2)
            v = tk.DoubleVar(value=val)
            self.pred_vars[feat] = v
            tk.Entry(c, textvariable=v, font=self.FONT, bg="#1e293b", fg=self.TEXT,
                     insertbackground=self.ACCENT, relief="flat", width=16).grid(
                row=i, column=1, padx=10, pady=2)

        tk.Label(c, text="Model:", font=self.FONT, fg=self.TEXT, bg=self.PANEL).grid(
            row=len(defaults), column=0, sticky="w", pady=6)
        self.pred_model_var = tk.StringVar(value="Random Forest")
        ttk.Combobox(c, textvariable=self.pred_model_var,
                     values=list(MODELS.keys()), state="readonly", width=18,
                     font=self.FONT).grid(row=len(defaults), column=1, padx=10)

        self._btn(c, "🔍  Classify Traffic", self._predict_single).grid(
            row=len(defaults)+1, column=0, columnspan=2, sticky="ew", pady=10)

        # Result card
        c2 = self._card(t, "🎯  Prediction Result", row=0, col=1, sticky="nsew")
        self.pred_result_var = tk.StringVar(value="—")
        tk.Label(c2, textvariable=self.pred_result_var,
                 font=("Consolas", 28, "bold"), fg=self.ACCENT, bg=self.PANEL,
                 wraplength=400, justify="center").pack(expand=True)
        self.pred_prob_var = tk.StringVar(value="")
        tk.Label(c2, textvariable=self.pred_prob_var,
                 font=self.FONT, fg=self.SUCCESS, bg=self.PANEL).pack()

    # ─── TAB 5 : PERFORMANCE ──────────────────────────────────
    def _build_perf_tab(self):
        t = self.tab_perf
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)
        tk.Label(t, text="Train models to view comparative performance charts.",
                 font=self.FONT, fg=self.SUBTEXT, bg=self.BG).pack(pady=40)

    # ══════════════════════════════════════════════════════════
    # 3.  ACTIONS
    # ══════════════════════════════════════════════════════════

    def _log(self, msg: str):
        self.log.config(state="normal")
        self.log.insert("end", f"  ▶  {msg}\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _set_status(self, text, color):
        self.status_var.set(text)
        self.status_lbl.config(fg=color)

    def _generate_dataset(self):
        def _work():
            self._set_status("● GENERATING", self.WARN)
            self._log(f"Generating {self.n_samples_var.get():,} samples …")
            self.df = generate_5g_dataset(self.n_samples_var.get())
            self._log(f"Dataset ready → shape {self.df.shape}  |  classes: {self.df['traffic_class'].nunique()}")
            self._log("Class distribution:\n" +
                      "\n".join(f"      {k}: {v}" for k, v in
                                self.df["traffic_class"].value_counts().items()))
            self.ds_info_var.set(
                f"✔  Generated  |  {self.df.shape[0]:,} rows  ×  {self.df.shape[1]} cols  |  "
                f"{self.df['traffic_class'].nunique()} classes"
            )
            self._set_status("● READY", self.SUCCESS)

        threading.Thread(target=_work, daemon=True).start()

    # ── CSV Load ───────────────────────────────────────────────
    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select 5G Traffic CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.load_path_var.set(path)

    def _load_csv(self):
        path = self.load_path_var.get()
        if not path or path == "No file selected":
            messagebox.showwarning("No File", "Please browse and select a CSV file first.")
            return

        def _work():
            global FEATURE_COLS
            self._set_status("● LOADING", self.WARN)
            self._log(f"Loading CSV: {path}")
            try:
                df = pd.read_csv(path)

                # ── Basic validation ──────────────────────────
                label_col = self.label_col_var.get().strip()
                if label_col not in df.columns:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Column Not Found",
                        f"Label column '{label_col}' not found.\n\n"
                        f"Available columns:\n{', '.join(df.columns.tolist())}"
                    ))
                    self._set_status("● ERROR", "#ef4444")
                    return

                # Rename label column to standard name if different
                if label_col != "traffic_class":
                    df = df.rename(columns={label_col: "traffic_class"})
                    self._log(f"  Renamed '{label_col}' → 'traffic_class'")

                # ── Auto-detect numeric feature columns ───────
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # Remove any index-like columns
                numeric_cols = [c for c in numeric_cols if c.lower() not in ("id", "index", "unnamed: 0")]

                missing_feats = [f for f in FEATURE_COLS if f not in numeric_cols]
                if missing_feats:
                    # Try to proceed with available numeric cols only
                    available = [f for f in FEATURE_COLS if f in df.columns]
                    if len(available) < 3:
                        self.root.after(0, lambda: messagebox.showerror(
                            "Incompatible CSV",
                            f"Could not find enough feature columns.\n\n"
                            f"Expected columns (any subset of):\n{', '.join(FEATURE_COLS)}\n\n"
                            f"Found numeric columns:\n{', '.join(numeric_cols)}"
                        ))
                        self._set_status("● ERROR", "#ef4444")
                        return
                    self._log(f"  ⚠ Missing features: {missing_feats}")
                    self._log(f"  Using available features: {available}")
                    # Patch global FEATURE_COLS for this session
                    FEATURE_COLS = available

                # Drop rows with NaN in feature or label columns
                before = len(df)
                df = df.dropna(subset=FEATURE_COLS + ["traffic_class"])
                dropped = before - len(df)
                if dropped:
                    self._log(f"  Dropped {dropped} rows with NaN values")

                # ── Convert feature cols to numeric ───────────
                for col in FEATURE_COLS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=FEATURE_COLS)

                self.df = df.reset_index(drop=True)

                classes   = self.df["traffic_class"].nunique()
                cls_dist  = self.df["traffic_class"].value_counts()

                self._log(f"  ✔ Loaded  →  {self.df.shape[0]:,} rows  ×  {self.df.shape[1]} cols")
                self._log(f"  Features : {FEATURE_COLS}")
                self._log(f"  Classes  : {classes}")
                self._log("  Class distribution:\n" +
                          "\n".join(f"      {k}: {v}" for k, v in cls_dist.items()))

                self.ds_info_var.set(
                    f"✔  Loaded CSV  |  {self.df.shape[0]:,} rows  ×  {self.df.shape[1]} cols  |  "
                    f"{classes} classes"
                )
                self._set_status("● READY", self.SUCCESS)

            except Exception as exc:
                self._log(f"  ✘ Error loading CSV: {exc}")
                self.root.after(0, lambda: messagebox.showerror("Load Error", str(exc)))
                self._set_status("● ERROR", "#ef4444")

        threading.Thread(target=_work, daemon=True).start()

    def _run_eda(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please generate or load a dataset first.")
            return
        threading.Thread(target=self._eda_plots, daemon=True).start()

    def _eda_plots(self):
        self._set_status("● EDA RUNNING", self.WARN)
        self._log("Running EDA …")

        df = self.df
        # Use actual classes present in dataset (works for both generated & loaded CSV)
        classes_in_df = sorted(df["traffic_class"].unique().tolist())
        n_cls = len(classes_in_df)

        # Clear old EDA tab content
        for w in self.tab_eda.winfo_children():
            w.destroy()

        # ── Create a scrollable canvas inside EDA tab ─────────
        canvas_outer = tk.Canvas(self.tab_eda, bg=self.BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self.tab_eda, orient="vertical", command=canvas_outer.yview)
        canvas_outer.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas_outer.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas_outer, bg=self.BG)
        win_id = canvas_outer.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(e):
            canvas_outer.configure(scrollregion=canvas_outer.bbox("all"))
            canvas_outer.itemconfig(win_id, width=canvas_outer.winfo_width())
        inner.bind("<Configure>", on_configure)

        plt.style.use("dark_background")
        colors = ["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444","#06b6d4","#8b5cf6"]

        # ── PLOT 1 : Class Distribution ────────────────────────
        fig1, ax = plt.subplots(figsize=(10, 4), facecolor="#111827")
        counts = df["traffic_class"].value_counts()
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="#0a0e1a", linewidth=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    color="#e2e8f0", fontsize=9, fontfamily="monospace")
        ax.set_facecolor("#0d1117"); ax.set_title("Traffic Class Distribution", color="#00d4ff", fontsize=13)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.set_xlabel("Traffic Class", color="#94a3b8")
        ax.set_ylabel("Count",         color="#94a3b8")
        plt.xticks(rotation=15)
        plt.tight_layout()
        self._embed_fig(fig1, inner, "📊  Class Distribution")

        # ── PLOT 2 : Feature Correlation Heatmap ──────────────
        fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor="#111827")
        corr = df[FEATURE_COLS].corr()
        sns.heatmap(corr, ax=ax2, cmap="coolwarm", annot=True, fmt=".2f",
                    annot_kws={"size": 8}, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        ax2.set_facecolor("#0d1117"); ax2.set_title("Feature Correlation Matrix", color="#00d4ff", fontsize=13)
        ax2.tick_params(colors="#94a3b8", labelsize=8)
        plt.tight_layout()
        self._embed_fig(fig2, inner, "🔥  Correlation Heatmap")

        # ── PLOT 3 : Feature Distributions by Class ───────────
        features_to_plot = [f for f in ["throughput_mbps", "latency_ms", "packet_loss_pct", "avg_packet_size_B"] if f in df.columns]
        if len(features_to_plot) >= 2:
            fig3, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#111827")
            for ax3, feat in zip(axes.flat, features_to_plot):
                for i, cls in enumerate(classes_in_df):
                    data = df[df["traffic_class"] == cls][feat]
                    ax3.hist(data, bins=30, alpha=0.5, color=colors[i % len(colors)], label=cls, density=True)
                ax3.set_facecolor("#0d1117")
                ax3.set_title(feat.replace("_", " ").title(), color="#00d4ff", fontsize=10)
                ax3.tick_params(colors="#94a3b8", labelsize=7)
            handles = [mpatches.Patch(color=colors[i], label=c) for i, c in enumerate(classes_in_df)]
            fig3.legend(handles=handles, loc="lower center", ncol=min(4, n_cls),
                        fontsize=8, facecolor="#111827", labelcolor="#e2e8f0")
            fig3.suptitle("Feature Distributions by Traffic Class", color="#00d4ff", fontsize=13)
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            self._embed_fig(fig3, inner, "📉  Feature Distributions")

        # ── PLOT 4 : Box Plots ────────────────────────────────
        box_feats = [f for f in ["throughput_mbps", "latency_ms", "jitter_ms"] if f in df.columns]
        if box_feats:
            fig4, axes4 = plt.subplots(1, len(box_feats), figsize=(14, 5), facecolor="#111827")
            if len(box_feats) == 1:
                axes4 = [axes4]
            for ax4, feat in zip(axes4, box_feats):
                data_by_class = [df[df["traffic_class"] == c][feat].values for c in classes_in_df]
                bp = ax4.boxplot(data_by_class, patch_artist=True,
                                 medianprops=dict(color="#00d4ff", linewidth=2))
                for patch, color in zip(bp["boxes"], colors * (n_cls // len(colors) + 1)):
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                ax4.set_facecolor("#0d1117")
                ax4.set_xticks(range(1, n_cls + 1))
                ax4.set_xticklabels([c.replace("_", "\n") for c in classes_in_df],
                                     color="#94a3b8", fontsize=7)
                ax4.set_title(feat.replace("_", " ").title(), color="#00d4ff", fontsize=10)
                ax4.tick_params(colors="#94a3b8")
            fig4.suptitle("Boxplot by Traffic Class", color="#00d4ff", fontsize=13)
            plt.tight_layout()
            self._embed_fig(fig4, inner, "📦  Box Plots")

        # ── PLOT 5 : Scatter — Throughput vs Latency ─────────
        if "throughput_mbps" in df.columns and "latency_ms" in df.columns:
            fig5, ax5 = plt.subplots(figsize=(9, 5), facecolor="#111827")
            for i, cls in enumerate(classes_in_df):
                sub = df[df["traffic_class"] == cls]
                ax5.scatter(sub["throughput_mbps"], sub["latency_ms"],
                            alpha=0.4, s=18, color=colors[i % len(colors)], label=cls)
            ax5.set_facecolor("#0d1117")
            ax5.set_xlabel("Throughput (Mbps)", color="#94a3b8")
            ax5.set_ylabel("Latency (ms)",      color="#94a3b8")
            ax5.set_title("Throughput vs Latency", color="#00d4ff", fontsize=13)
            ax5.legend(fontsize=8, facecolor="#111827", labelcolor="#e2e8f0")
            ax5.tick_params(colors="#94a3b8")
            plt.tight_layout()
            self._embed_fig(fig5, inner, "🔵  Scatter: Throughput vs Latency")

        # ── PLOT 6 : Pairplot subset ───────────────────────────
        pp_cols = [f for f in ["throughput_mbps","latency_ms","jitter_ms","packet_loss_pct"] if f in df.columns]
        if len(pp_cols) >= 2:
            self._log("Generating pairplot (may take a moment) …")
            n_pp = min(700, len(df))
            subset = df[pp_cols + ["traffic_class"]].sample(n_pp)
            palette = {c: colors[i % len(colors)] for i, c in enumerate(classes_in_df)}
            pp = sns.pairplot(subset, hue="traffic_class", palette=palette,
                              plot_kws={"alpha": 0.4, "s": 14},
                              diag_kws={"fill": True, "alpha": 0.5})
            pp.figure.set_facecolor("#111827")
            for ax_pp in pp.axes.flat:
                ax_pp.set_facecolor("#0d1117")
                ax_pp.tick_params(colors="#94a3b8")
            pp.figure.suptitle("Pairplot: Key Features", y=1.01, color="#00d4ff", fontsize=12)
            self._embed_fig(pp.figure, inner, "🔗  Pairplot")

        self._log("EDA complete.")
        self._set_status("● READY", self.SUCCESS)

    def _embed_fig(self, fig, parent, title=""):
        frame = tk.Frame(parent, bg=self.PANEL, bd=0,
                         highlightbackground=self.ACCENT2, highlightthickness=1)
        frame.pack(fill="x", padx=14, pady=10)
        if title:
            tk.Label(frame, text=title, font=self.FONT_B,
                     fg=self.ACCENT, bg=self.PANEL).pack(anchor="w", padx=10, pady=(8, 0))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=6, pady=6)
        plt.close(fig)

    # ── TRAINING ───────────────────────────────────────────────
    def _train_models(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Generate or load a dataset first.")
            return
        threading.Thread(target=self._train_work, daemon=True).start()

    def _train_work(self):
        self._set_status("● TRAINING", self.WARN)
        df = self.df

        X = df[FEATURE_COLS].values
        y = self.le.fit_transform(df["traffic_class"].values)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size_var.get(), random_state=42, stratify=y
        )
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s  = self.scaler.transform(X_te)

        self.results = {}
        for name, model in MODELS.items():
            self._log(f"Training {name} …")
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            acc  = accuracy_score(y_te, y_pred)
            cv   = cross_val_score(model, X_tr_s, y_tr, cv=5, scoring="accuracy")
            cm   = confusion_matrix(y_te, y_pred)
            cr   = classification_report(y_te, y_pred,
                                          target_names=self.le.classes_, output_dict=True)
            try:
                y_prob = model.predict_proba(X_te_s)
                auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
            except Exception:
                auc = None

            self.results[name] = dict(acc=acc, cv=cv, cm=cm, cr=cr, auc=auc, model=model)
            self.trained[name] = model
            self._log(f"  {name}: acc={acc:.4f}  cv={cv.mean():.4f}±{cv.std():.4f}"
                      + (f"  AUC={auc:.4f}" if auc else ""))

        self._update_train_tab()
        self._log("All models trained ✔")
        self._set_status("● READY", self.SUCCESS)

    def _update_train_tab(self):
        for w in self.tab_train.winfo_children():
            w.destroy()

        t = self.tab_train
        t.columnconfigure((0,1,2,3), weight=1)

        headers = ["Model", "Accuracy", "CV Mean ± Std", "AUC (macro)"]
        for j, h in enumerate(headers):
            tk.Label(t, text=h, font=self.FONT_B, fg=self.ACCENT,
                     bg=self.PANEL, padx=12, pady=6, relief="flat").grid(
                row=0, column=j, sticky="ew", padx=4, pady=4)

        for i, (name, r) in enumerate(self.results.items(), start=1):
            bg = self.PANEL if i % 2 == 0 else "#161e2e"
            vals = [
                name,
                f"{r['acc']:.4f}",
                f"{r['cv'].mean():.4f} ± {r['cv'].std():.4f}",
                f"{r['auc']:.4f}" if r['auc'] else "N/A",
            ]
            for j, v in enumerate(vals):
                try:
                    fg = self.SUCCESS if j == 1 and float(v) > 0.9 else self.TEXT
                except ValueError:
                    fg = self.TEXT
                tk.Label(t, text=v, font=self.FONT, fg=fg,
                         bg=bg, padx=12, pady=8).grid(
                    row=i, column=j, sticky="ew", padx=4, pady=2)

        # Confusion matrices
        frame_cm = tk.Frame(t, bg=self.BG)
        frame_cm.grid(row=len(self.results)+1, column=0, columnspan=4,
                      sticky="nsew", padx=8, pady=12)
        t.rowconfigure(len(self.results)+1, weight=1)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#111827")
        for ax, (name, r) in zip(axes, self.results.items()):
            sns.heatmap(r["cm"], ax=ax, cmap="Blues", annot=True, fmt="d",
                        xticklabels=self.le.classes_, yticklabels=self.le.classes_,
                        annot_kws={"size": 7}, cbar=False, linewidths=0.3)
            ax.set_facecolor("#0d1117")
            ax.set_title(name, color="#00d4ff", fontsize=10)
            ax.tick_params(colors="#94a3b8", labelsize=6)
            ax.set_xlabel("Predicted", color="#94a3b8", fontsize=8)
            ax.set_ylabel("Actual",    color="#94a3b8", fontsize=8)
        fig.suptitle("Confusion Matrices", color="#00d4ff", fontsize=13)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame_cm)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ── PREDICT ────────────────────────────────────────────────
    def _predict_single(self):
        if not self.trained:
            messagebox.showwarning("Not trained", "Train models first.")
            return
        vals = np.array([[self.pred_vars[f].get() for f in FEATURE_COLS]])
        vals_s = self.scaler.transform(vals)
        model_name = self.pred_model_var.get()
        model = self.trained[model_name]
        pred = model.predict(vals_s)[0]
        label = self.le.inverse_transform([pred])[0]
        self.pred_result_var.set(label.replace("_", " "))
        try:
            prob = model.predict_proba(vals_s)[0][pred]
            self.pred_prob_var.set(f"Confidence: {prob*100:.1f}%  |  Model: {model_name}")
        except Exception:
            self.pred_prob_var.set(f"Model: {model_name}")

    # ── PERFORMANCE PLOTS ──────────────────────────────────────
    def _plot_perf(self):
        if not self.results:
            messagebox.showwarning("No results", "Train models first.")
            return
        threading.Thread(target=self._perf_plots, daemon=True).start()

    def _perf_plots(self):
        for w in self.tab_perf.winfo_children():
            w.destroy()

        plt.style.use("dark_background")
        colors = ["#00d4ff","#7c3aed","#10b981","#f59e0b"]
        names  = list(self.results.keys())

        # ── Canvas wrapper ────────────────────────────────────
        canvas_outer = tk.Canvas(self.tab_perf, bg=self.BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self.tab_perf, orient="vertical", command=canvas_outer.yview)
        canvas_outer.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas_outer.pack(fill="both", expand=True)
        inner = tk.Frame(canvas_outer, bg=self.BG)
        win_id = canvas_outer.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas_outer.configure(
            scrollregion=canvas_outer.bbox("all")))

        # ── Accuracy comparison ────────────────────────────────
        fig1, ax = plt.subplots(figsize=(10, 4), facecolor="#111827")
        accs = [self.results[n]["acc"] for n in names]
        cvs  = [self.results[n]["cv"].mean() for n in names]
        x    = np.arange(len(names))
        ax.bar(x-0.18, accs, width=0.35, label="Test Acc",   color=colors[0], alpha=0.9)
        ax.bar(x+0.18, cvs,  width=0.35, label="CV Mean Acc",color=colors[1], alpha=0.9)
        ax.set_facecolor("#0d1117")
        ax.set_xticks(x); ax.set_xticklabels(names, color="#94a3b8", fontsize=9)
        ax.set_ylim(0.5, 1.05)
        ax.set_title("Model Accuracy Comparison", color="#00d4ff", fontsize=13)
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#111827", labelcolor="#e2e8f0")
        for xi, a, c in zip(x, accs, cvs):
            ax.text(xi-0.18, a+0.005, f"{a:.3f}", ha="center", va="bottom",
                    color="#e2e8f0", fontsize=8, fontfamily="monospace")
            ax.text(xi+0.18, c+0.005, f"{c:.3f}", ha="center", va="bottom",
                    color="#e2e8f0", fontsize=8, fontfamily="monospace")
        plt.tight_layout()
        self._embed_fig(fig1, inner, "🏆  Accuracy Comparison")

        # ── CV Box plots ───────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor="#111827")
        cv_data = [self.results[n]["cv"] for n in names]
        bp = ax2.boxplot(cv_data, patch_artist=True,
                         medianprops=dict(color="#00d4ff", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax2.set_facecolor("#0d1117")
        ax2.set_xticks(range(1, len(names)+1))
        ax2.set_xticklabels(names, color="#94a3b8", fontsize=9)
        ax2.set_title("Cross-Validation Score Distribution (5-fold)", color="#00d4ff", fontsize=13)
        ax2.tick_params(colors="#94a3b8")
        plt.tight_layout()
        self._embed_fig(fig2, inner, "📦  CV Score Distribution")

        # ── Per-class F1 ───────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(12, 5), facecolor="#111827")
        classes = list(self.le.classes_)
        x_c = np.arange(len(classes))
        width = 0.2
        for j, (n, clr) in enumerate(zip(names, colors)):
            f1s = [self.results[n]["cr"][c]["f1-score"] for c in classes]
            ax3.bar(x_c + j*width - 0.3, f1s, width, label=n, color=clr, alpha=0.85)
        ax3.set_facecolor("#0d1117")
        ax3.set_xticks(x_c); ax3.set_xticklabels(classes, color="#94a3b8", fontsize=8, rotation=15)
        ax3.set_title("Per-Class F1 Score by Model", color="#00d4ff", fontsize=13)
        ax3.tick_params(colors="#94a3b8")
        ax3.legend(facecolor="#111827", labelcolor="#e2e8f0", fontsize=9)
        plt.tight_layout()
        self._embed_fig(fig3, inner, "🎯  Per-Class F1 Score")

        # ── AUC comparison ────────────────────────────────────
        auc_names = [n for n in names if self.results[n]["auc"]]
        if auc_names:
            fig4, ax4 = plt.subplots(figsize=(7, 4), facecolor="#111827")
            aucs = [self.results[n]["auc"] for n in auc_names]
            bars = ax4.barh(auc_names, aucs, color=colors[:len(auc_names)], alpha=0.85)
            ax4.set_facecolor("#0d1117")
            ax4.set_xlim(0.5, 1.05)
            ax4.set_title("AUC-ROC (macro OvR)", color="#00d4ff", fontsize=13)
            ax4.tick_params(colors="#94a3b8")
            for bar, v in zip(bars, aucs):
                ax4.text(v+0.003, bar.get_y()+bar.get_height()/2,
                         f"{v:.4f}", va="center", color="#e2e8f0",
                         fontsize=9, fontfamily="monospace")
            plt.tight_layout()
            self._embed_fig(fig4, inner, "📡  AUC-ROC Score")

        # ── Feature Importance (Random Forest) ────────────────
        if "Random Forest" in self.results:
            rf = self.results["Random Forest"]["model"]
            fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
            fig5, ax5 = plt.subplots(figsize=(9, 5), facecolor="#111827")
            fi.plot(kind="barh", ax=ax5, color="#7c3aed", alpha=0.85)
            ax5.set_facecolor("#0d1117")
            ax5.set_title("Feature Importance — Random Forest", color="#00d4ff", fontsize=13)
            ax5.tick_params(colors="#94a3b8", labelsize=9)
            ax5.set_xlabel("Importance", color="#94a3b8")
            plt.tight_layout()
            self._embed_fig(fig5, inner, "🌲  Feature Importance")

        self._log("Performance plots generated ✔")
        self._set_status("● READY", self.SUCCESS)

    # ── SAVE MODELS ────────────────────────────────────────────
    def _save_models(self):
        if not self.trained:
            messagebox.showwarning("Nothing to save", "Train models first.")
            return
        for name, model in self.trained.items():
            fname = name.lower().replace(" ", "_") + "_model.pkl"
            joblib.dump(model, fname)
        joblib.dump(self.scaler, "scaler.pkl")
        joblib.dump(self.le,     "label_encoder.pkl")
        self._log("Models saved: " + ", ".join(
            n.lower().replace(" ","_")+"_model.pkl" for n in self.trained))
        messagebox.showinfo("Saved", "All models saved to disk (.pkl files).")


# ══════════════════════════════════════════════════════════════
# 4.  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = TrafficClassifierApp(root)
    root.mainloop()
