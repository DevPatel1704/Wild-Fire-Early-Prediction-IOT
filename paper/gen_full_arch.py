"""Full WildfireEWS system architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG   = "#07101a"; TEXT = "#e6edf3"; MUT  = "#7a9aaa"; LINE = "#1d2a35"
C_DATA="#1e4060"; C_KAFKA="#2d1f40"; C_PRE="#1e3530"; C_GAT="#0d2a2a"
TEAL="#2dd4bf"; AMB="#f59e0b"; PURP="#a78bfa"; RED="#ef4444"
BLUE="#60a5fa"; GRN="#34d399"; ORG="#f97316"

def rbox(ax, xc, yc, w, h, fc, ec, lw=1.2, alpha=1.0):
    r = FancyBboxPatch((xc-w/2, yc-h/2), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(r)

def t(ax, x, y, s, size=8, color=TEXT, bold=False, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontfamily="DejaVu Sans Mono",
            fontweight="bold" if bold else "normal", zorder=4)

def arr(ax, x0, y0, x1, y1, color=TEAL, lw=1.5, rad=0.0, label=""):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)
    if label:
        ax.text((x0+x1)/2+0.12, (y0+y1)/2+0.08, label, fontsize=6,
                color=color, fontfamily="DejaVu Sans Mono", zorder=6, ha="center")

def sec(ax, x, y, w, h, color, label):
    r = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor=color, linewidth=0, alpha=0.13, zorder=1)
    ax.add_patch(r)
    ax.text(x+0.18, y+h-0.22, label, fontsize=7.5, color=color,
            fontfamily="DejaVu Sans Mono", fontweight="bold", alpha=0.9, zorder=2)

fig, ax = plt.subplots(figsize=(20,14))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0,20); ax.set_ylim(0,14); ax.axis("off")

t(ax,10,13.55,"WildfireEWS  --  Full System Architecture",size=17,bold=True)
t(ax,10,13.15,"Real-Time Wildfire Early Warning System  |  IoT + GAT-LSTM + FastAPI + React Dashboard",size=9,color=MUT)

sec(ax, 0.2,10.2, 4.0,2.5, BLUE, "[1] DATA SOURCES")
sec(ax, 4.5,10.2, 3.2,2.5, PURP, "[2] KAFKA INGESTION")
sec(ax, 0.2, 6.8, 7.5,3.0, GRN,  "[3] FEATURE ENGINEERING & PREPROCESSING")
sec(ax, 8.0, 6.0, 5.8,6.7, TEAL, "[4] MODEL TRAINING")
sec(ax,14.0, 6.0, 5.8,6.7, AMB,  "[5] REAL-TIME INFERENCE & DASHBOARD")
sec(ax, 0.2, 0.4,19.6,5.6, ORG,  "[6] EVALUATION & REPORTING")

# --- [1] DATA SOURCES ---
for xc, yc, title, sub, col in [
    (1.1,12.1,"ERA5 NetCDF",   "Climate reanalysis\nBC bounding box\nDaily resolution", BLUE),
    (2.3,12.1,"NFDB CSV",      "National Fire DB\nHistorical fire pts\n1950-2023",       BLUE),
    (3.5,12.1,"NASA FIRMS",    "Active fire CSV\nMODIS/VIIRS\nNear real-time",           BLUE),
    (1.7,11.0,"Synthetic Gen", "Physics-based\n730 days  16x16 grid\nseed=3",            C_DATA),
]:
    rbox(ax,xc,yc,1.1,0.72,col,BLUE,lw=1.0)
    t(ax,xc,yc+0.16,title,size=7.5,bold=True)
    t(ax,xc,yc-0.13,sub,size=6.0,color=MUT)
arr(ax,4.2,11.5,4.5,11.5,BLUE,lw=2.0)

# --- [2] KAFKA ---
rbox(ax,5.5,12.0,2.5,0.75,C_KAFKA,PURP,lw=1.3)
t(ax,5.5,12.20,"Apache Kafka",size=8.5,bold=True,color=PURP)
t(ax,5.5,11.95,"Topic: sensor.readings",size=7,color=MUT)
t(ax,5.5,11.72,"at-least-once  replay  buffering",size=6.5,color=MUT)
rbox(ax,5.5,11.0,2.5,0.55,C_KAFKA,PURP,lw=0.9)
t(ax,5.5,11.00,"Topic: predictions",size=7.5,bold=True,color=PURP)
t(ax,5.5,10.78,"GAT-LSTM + XGBoost outputs",size=6.2,color=MUT)
arr(ax,5.5,11.62,5.5,11.28,PURP,lw=1.2,label="publish")

# --- [3] PREPROCESSING ---
for xc, yc, title, sub in [
    (1.2,9.5,"Feature\nSelection","Top-10 ERA5\nPaper Table 9"),
    (2.6,9.5,"Spatial\nSnapping", "16x16->10x10\nNearest node"),
    (4.0,9.5,"Sliding\nWindow",   "7-day sequences\n[S,T,N,F]"),
    (5.4,9.5,"70:30\nChron Split","Train/Test\nno leakage"),
    (6.8,9.5,"RUS +\nMin-Max",    "Balance train\nScale features"),
]:
    rbox(ax,xc,yc,1.15,0.80,C_PRE,GRN,lw=1.0)
    t(ax,xc,yc+0.18,title,size=7.5,bold=True)
    t(ax,xc,yc-0.14,sub,size=6.0,color=MUT)
for xi in [1.78,3.18,4.58,5.98]:
    arr(ax,xi,9.5,xi+0.25,9.5,GRN,lw=1.2)
arr(ax,2.2,11.0,2.2,9.9,BLUE,lw=1.4,rad=0.1)
arr(ax,5.5,10.72,5.5,9.9,PURP,lw=1.2)

rbox(ax,3.8,7.6,3.2,0.85,C_PRE,GRN,lw=1.1)
t(ax,3.8,7.85,"IoT Sensor Graph",size=8.5,bold=True,color=GRN)
t(ax,3.8,7.62,"100 nodes  10x10 grid over BC",size=7,color=MUT)
t(ax,3.8,7.40,"k-NN adjacency (k=4)  Adj matrix [100x100]",size=6.5,color=MUT)
for ri in range(4):
    for ci in range(4):
        nx=1.0+ci*0.22; ny=7.35+ri*0.22
        ax.plot(nx,ny,'o',color=(TEAL if (ri+ci)%3==0 else MUT),markersize=4,zorder=4)
        if ci<3: ax.plot([nx,nx+0.22],[ny,ny],'-',color=LINE,lw=0.7,zorder=3)
        if ri<3: ax.plot([nx,nx],[ny,ny+0.22],'-',color=LINE,lw=0.7,zorder=3)
ax.text(1.4,7.22,"sensor grid",fontsize=6,color=MUT,fontfamily="DejaVu Sans Mono",ha="center")
arr(ax,4.0,9.1,4.0,8.03,GRN,lw=1.3)

# --- [4] MODEL TRAINING ---
rbox(ax,10.6,11.8,5.0,1.5,C_GAT,TEAL,lw=1.8)
t(ax,10.6,12.35,"GAT-LSTM  [PRIMARY MODEL]",size=9.5,bold=True,color=TEAL)
for xc,yc,lab,sub,col in [
    (8.9, 11.75,"GAT Layer 1","in=10->64 h=4\nconcat->[B,N,256]",TEAL),
    (10.1,11.75,"GAT Layer 2","in=256->64 h=1\nmean->[B,N,64]",  TEAL),
    (11.3,11.75,"LayerNorm",  "[B,N,64]\nstabilise",             GRN),
    (12.5,11.75,"LSTM",       "hidden=128 2L\n[B*N,T,128]",      PURP),
    (13.3,11.75,"Head",       "128->64 GELU\n->Drop->1",         RED),
]:
    rbox(ax,xc,yc,1.05,0.70,"#0a1a22",col,lw=0.9)
    t(ax,xc,yc+0.16,lab,size=7,bold=True,color=col)
    t(ax,xc,yc-0.14,sub,size=5.8,color=MUT)
for xi in [9.43,10.63,11.83,12.83]:
    arr(ax,xi,11.75,xi+0.22,11.75,TEAL,lw=1.0)
rbox(ax,10.6,10.55,5.0,0.95,"#0a1520",TEAL,lw=1.0)
t(ax,10.6,10.88,"Training Config",size=8,bold=True,color=TEAL)
t(ax,10.6,10.65,"Adam lr=1e-3  weight_decay=1e-4  CosineAnnealingLR  grad_clip=1.0",size=6.5,color=MUT)
t(ax,10.6,10.42,"BCEWithLogitsLoss+pos_weight  batch=32  max_epochs=50  early_stop F2 patience=10",size=6.5,color=MUT)
arr(ax,10.6,11.05,10.6,11.4,TEAL,lw=1.2)

t(ax,11.0,10.15,"Baselines (Paper Table 4 hyperparameters)",size=8,bold=True,color=MUT)
for xc,yc,lab,sub,col in [
    (8.6, 9.5,"RNN+LSTM",    "2x64 LSTM\nAdam lr=0.001\nbatch=128",PURP),
    (9.8, 9.5,"CatBoost",    "iter=500\nlr=0.05 d=6\nLogloss",     AMB),
    (11.0,9.5,"RandomForest","n_est=100\ndepth=10\nbalanced",       GRN),
    (12.2,9.5,"XGBoost",     "n_est=200\nlr=0.1 d=6\nsub=0.8",     BLUE),
    (13.4,9.5,"LightGBM",    "leaves=200\nlr=0.05\nn_est=100",      ORG),
]:
    rbox(ax,xc,yc,1.05,0.85,"#0a1520",col,lw=0.9)
    t(ax,xc,yc+0.22,lab,size=7,bold=True,color=col)
    t(ax,xc,yc-0.08,sub,size=5.8,color=MUT)
arr(ax,7.5,9.0, 8.4,9.9, GRN,lw=1.4,rad=-0.1)
arr(ax,7.5,9.0,10.6,10.5,GRN,lw=1.4,rad=-0.15,label="train tensors [S,T,N,F]")

rbox(ax,10.6,8.45,5.0,0.80,"#0a1a14",GRN,lw=1.0)
t(ax,10.6,8.72,"Saved Artifacts",size=8,bold=True,color=GRN)
t(ax,10.6,8.50,"gat_lstm.pt  xgb.json  scaler.json  graph.json  model_results.json  train_history.json",size=6.2,color=MUT)
for xc in [8.8+i for i in range(6)]:
    arr(ax,xc,9.08,xc,8.85,GRN,lw=0.8)

# --- [5] REAL-TIME INFERENCE ---
rbox(ax,16.5,11.8,2.8,1.4,"#1a1a30",PURP,lw=1.3)
t(ax,16.5,12.28,"Stream Predictor",size=9,bold=True,color=PURP)
t(ax,16.5,12.02,"Loads gat_lstm.pt + xgb.json",size=7,color=MUT)
t(ax,16.5,11.80,"GAT-LSTM forecast  every 7 min",size=7,color=TEAL)
t(ax,16.5,11.60,"XGBoost alert       every 60 s",size=7,color=AMB)
t(ax,16.5,11.40,"-> publish to predictions topic",size=6.5,color=MUT)
rbox(ax,16.5,10.2,2.8,1.1,"#1a0a2a",PURP,lw=1.3)
t(ax,16.5,10.58,"FastAPI Backend",size=9,bold=True,color=PURP)
t(ax,16.5,10.35,"GET /nodes   GET /alerts",size=7,color=MUT)
t(ax,16.5,10.15,"GET /eval-results   WS /ws",size=7,color=MUT)
t(ax,16.5, 9.95,"StaticFiles /eval/*.png",size=7,color=MUT)
arr(ax,16.5,11.1,16.5,10.75,PURP,lw=1.3)
rbox(ax,16.5,8.85,2.8,1.4,"#0a1e2a",TEAL,lw=1.5)
t(ax,16.5,9.33,"React Dashboard",size=9,bold=True,color=TEAL)
t(ax,16.5,9.10,"Tab 1: GAT-LSTM  live risk map",size=7,color=TEAL)
t(ax,16.5,8.90,"Tab 2: XGBoost  alert feed",size=7,color=AMB)
t(ax,16.5,8.68,"Tab 3: Results  21 eval charts",size=7,color=PURP)
t(ax,16.5,8.48,"WebSocket live  Vite proxy :3001",size=6.5,color=MUT)
arr(ax,16.5,9.65,16.5,9.55,TEAL,lw=1.3)
arr(ax,13.1,8.45,15.1,11.4,AMB,lw=1.4,rad=0.25,label="load weights")
arr(ax,7.5,10.72,15.1,11.65,PURP,lw=1.2,rad=-0.15,label="consume sensor.readings")

# --- [6] EVALUATION ---
t(ax,10.0,5.0,"Evaluation Report  --  21 Charts  -->  artifacts/eval/report.html",size=9,bold=True,color=ORG)
for xc,yc,lab,sub,col in [
    (1.4, 3.8,"ROC Curve",      "AUC=96.9%",        TEAL),
    (2.7, 3.8,"PR Curve",       "Avg Prec",          TEAL),
    (4.0, 3.8,"Confusion Mtrx", "F1-optimal thr",    TEAL),
    (5.3, 3.8,"Thresh Sweep",   "P/R vs thr",        AMB),
    (6.6, 3.8,"Calibration",    "Reliability",       AMB),
    (7.9, 3.8,"Prec Deep-Dive", "False alarm rate",  RED),
    (9.2, 3.8,"Node Heatmap",   "10x10 F1 grid",     GRN),
    (10.5,3.8,"Model Compare",  "6-model bars",      PURP),
    (11.8,3.8,"Metrics Heatmap","RdYlGn table",      PURP),
    (13.1,3.8,"ROC All Models", "6-model overlay",   BLUE),
    (14.4,3.8,"Train History",  "Loss F2 AUC",       ORG),
    (15.7,3.8,"Feature Import.","RF importance",     GRN),
    (17.0,3.8,"Class Balance",  "Fire rate 9.8%",    BLUE),
    (18.3,3.8,"Seasonality",    "DOY fire curve",    ORG),
]:
    rbox(ax,xc,yc,1.10,0.88,"#0a1218",col,lw=0.9)
    t(ax,xc,yc+0.22,lab,size=6.8,bold=True,color=col)
    t(ax,xc,yc-0.12,sub,size=6.0,color=MUT)

rbox(ax,4.85,2.25+len([1]*6)*0.32/2+0.9,6.5,len([1]*6)*0.32+1.0,"#0a1218",ORG,lw=1.0)
hdrs=["Model","Recall","F2 (*)","F1","Precision","AUC"]
col_x=[2.0,3.5,4.5,5.5,6.6,7.7]
row_y0=2.25
for ci,(h,cx) in enumerate(zip(hdrs,col_x)):
    t(ax,cx,row_y0+len([1]*6)*0.32+0.52,h,size=7.5,bold=True,color=ORG)
for ri,(name,rec,f2,f1,prec,auc,col,primary) in enumerate([
    ("GAT-LSTM","96.7%","92.7%","87.1%","79.2%","96.9%",TEAL,True),
    ("RNN+LSTM","96.9%","92.6%","87.0%","78.9%","96.8%",RED, False),
    ("RF",      "95.3%","92.4%","88.3%","82.3%","97.1%",GRN, False),
    ("CatBoost","95.4%","92.4%","88.2%","82.0%","97.2%",AMB, False),
    ("XGBoost", "94.9%","92.1%","88.2%","82.4%","97.1%",BLUE,False),
    ("LightGBM","94.4%","91.8%","88.3%","82.9%","97.1%",ORG, False),
]):
    ry=row_y0+(5-ri)*0.32
    if primary: rbox(ax,4.85,ry,6.3,0.28,col,col,lw=0.5,alpha=0.18)
    for ci,(v,cx) in enumerate(zip([name,rec,f2,f1,prec,auc],col_x)):
        c=col if ci==0 else (TEAL if ci in [1,2] and primary else TEXT)
        t(ax,cx,ry,v,size=7,color=c,bold=(ci==0 and primary))

arr(ax,10.6,8.05,10.6,5.1,ORG,lw=1.6,label="evaluate + plot")

# Legend
t(ax,17.5,7.5,"Legend",size=8,bold=True,color=MUT)
for i,(col,label) in enumerate([
    (TEAL,"GAT-LSTM / data flow"),
    (PURP,"Kafka / API"),
    (GRN, "Preprocessing / eval"),
    (AMB, "Training config / XGBoost"),
    (RED, "Loss / error"),
    (BLUE,"Data sources"),
    (ORG, "Evaluation results"),
]):
    yi=7.1-i*0.38
    ax.plot([16.3,16.9],[yi,yi],color=col,lw=2.2,zorder=5)
    t(ax,18.2,yi,label,size=7,color=col,ha="left")

fig.savefig("full_system_arch.png",dpi=150,bbox_inches="tight",facecolor=BG,edgecolor="none")
plt.close(fig)
print("Saved: full_system_arch.png")
