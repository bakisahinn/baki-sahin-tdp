import argparse, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

class LateFusionTabular(nn.Module):
    def __init__(self, clin_in, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(clin_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, hidden), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden*2, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, phys, clinical):
        h,_ = self.lstm(phys)
        c = self.mlp(clinical)
        return self.fusion(torch.cat([h,c], dim=-1)).squeeze(-1)

def main(csv_path):
    df=pd.read_csv(csv_path).sort_values(["patient_id","measurement_index"]).reset_index(drop=True)

    # Strict data checks from the supplied project method.
    if df["pain_class"].notna().any():
        print("pain_class is present; classification can be implemented only with labels defined by the project protocol.")
    else:
        print("pain_class is entirely empty: this run uses the explicitly supported NRS regression task only.")
    if df.groupby("patient_id")["split"].nunique().max()!=1:
        raise ValueError("Patient-level split violated.")
    if not df.groupby("patient_id").size().eq(14).all():
        raise ValueError("The dataset does not have exactly 14 measurements per patient.")

    # The CSV contains only paths for facial/ECG modalities. Do not fabricate files.
    ecg_ok=sum(os.path.exists(str(p)) for p in df["ecg_path"])
    face_ok=sum(os.path.exists(str(p)) for p in df["facial_data_path"])
    print(f"Resolvable ECG paths: {ecg_ok}/{len(df)}")
    print(f"Resolvable facial paths: {face_ok}/{len(df)}")
    if ecg_ok==0 or face_ok==0:
        print("Full multimodal CNN+LSTM model cannot be trained from this CSV alone.")

    # Current executable prototype: physiological LSTM + clinical/SF-MPQ MLP.
    phys_cols=["systolic_bp","diastolic_bp","heart_rate"]
    sf_cols=[c for c in df.columns if c.startswith("sfmpq_")]
    clinical_num=["age","asa","medication_use","medication_half_life","medication_effect_duration","last_medication_time"]

    w=df.copy()
    for c in ["medication_half_life","medication_effect_duration","last_medication_time"]:
        w[c]=w[c].fillna(0.0)
    w["sex_male"]=(w["sex"].str.lower()=="male").astype(float)
    w["sex_female"]=(w["sex"].str.lower()=="female").astype(float)
    for c in phys_cols:
        w[c+"_norm"]=w.groupby("patient_id")[c].transform(
            lambda s:(s-s.mean())/(s.std(ddof=0) if s.std(ddof=0)>0 else 1.0)
        )
    phys_norm=[c+"_norm" for c in phys_cols]
    clinical_cols=clinical_num+["sex_male","sex_female"]+sf_cols
    tr=w["split"].eq("train")
    means=w.loc[tr,clinical_cols].mean()
    stds=w.loc[tr,clinical_cols].std().replace(0,1).fillna(1)
    w[clinical_cols]=(w[clinical_cols]-means)/stds

    data={}
    for pid,g in w.groupby("patient_id",sort=True):
        g=g.sort_values("measurement_index")
        data[pid]=(
            torch.tensor(g[phys_norm].to_numpy(np.float32)).unsqueeze(0),
            torch.tensor(g[clinical_cols].to_numpy(np.float32)).unsqueeze(0),
            torch.tensor(g["nrs"].to_numpy(np.float32)).unsqueeze(0),
        )

    model=LateFusionTabular(len(clinical_cols))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_fn=nn.MSELoss()
    train_p=sorted(w.loc[w.split=="train","patient_id"].unique())
    val_p=sorted(w.loc[w.split=="validation","patient_id"].unique())
    test_p=sorted(w.loc[w.split=="test","patient_id"].unique())

    def run(pids,train=False):
        model.train(train); total=0; ys=[]; ps=[]
        for pid in pids:
            P,C,Y=data[pid]
            pred=model(P,C); loss=loss_fn(pred,Y)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
            total += float(loss.item()); ys.append(Y.squeeze().detach().numpy()); ps.append(pred.squeeze().detach().numpy())
        return total/len(pids),np.concatenate(ys),np.concatenate(ps)

    best=float("inf"); state=None; bad=0
    for ep in range(1,401):
        run(train_p,True)
        vl,_,_=run(val_p,False)
        if vl<best-1e-6:
            best=vl; state={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=40: break
    model.load_state_dict(state)
    _,y,p=run(test_p,False)
    print("Test MAE:",mean_absolute_error(y,p))
    print("Test RMSE:",mean_squared_error(y,p)**0.5)
    print("Test R2:",r2_score(y,p))
    print("Test Pearson r:",pearsonr(y,p).statistic)
    print("Test Spearman r:",spearmanr(y,p).statistic)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args=parser.parse_args()
    main(args.csv_path)
