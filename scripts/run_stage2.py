"""Portable, fail-fast second-stage queue. Run with the training environment's Python."""
import ctypes,datetime,json,os,subprocess,sys,time
from pathlib import Path
PR=Path(__file__).resolve().parent.parent
PY=Path(sys.executable)
STATE=PR/'results'/'stage2_queue.json'
records=[]
def run(root, name, args, expected=None, optional=False):
    if expected and all((root / p).exists() for p in expected):
        print("SKIP", name, flush=True)
        records.append(dict(name=name,status="reused"))
        save()
        return True
    logs = root / "results" / "logs"
    logs.mkdir(parents=True,exist_ok=True)
    log = logs / ("completion_" + name + ".log")
    command = [str(PY), "-u", *args]
    start = time.time()
    print("START", name, datetime.datetime.now().isoformat(), flush=True)
    with log.open("w",encoding="utf-8") as f:
        f.write(json.dumps(command,ensure_ascii=False)+"\n"); f.flush()
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT,env=env)
    ok = result.returncode == 0 and (not expected or all((root / p).exists() for p in expected))
    records.append(dict(name=name,status="complete" if ok else "failed",exit_code=result.returncode,
                        seconds=round(time.time()-start,3),command=command,log=str(log)))
    save()
    print("END",name,result.returncode,records[-1]["seconds"],flush=True)
    if not ok:
        print(log.read_text(encoding="utf-8",errors="replace")[-5000:],flush=True)
        if not optional: raise SystemExit("Failed: "+name)
    return ok
def save():
    STATE.write_text(json.dumps(dict(updated=datetime.datetime.now().isoformat(),records=records),
                               ensure_ascii=False,indent=2),encoding="utf-8")
def project():
    def ev(name,*args):
        return run(PR,"eval_"+name,["src/evaluate.py","--name",name,*args],
                   ["results/eval/"+name+".json","results/eval/"+name+"_pred.jsonl"])
    def adapt(name,*args):
        return run(PR,"adapt_"+name,["src/adapt_domain.py","--name",name,"--steps","200","--eval-every","25",
                   "--mem-frac","0.55",*args],["models/"+name+"/history.json","models/"+name+"/encoder.pt"])
    final=["--real","--oos","60","--real-index","--reject"]
    ck=["--grad-ckpt"]
    for name in ["tfidf","bm25"]:
        ev(name,"--baseline",name)
    ev("bm25_real","--baseline","bm25","--real-index","--reject")
    for name in ["raw_roberta","raw_rbt3"]:
        ev(name,"--baseline",name)
    models=["roberta_seed1","roberta_seed2","roberta_seed3","rbt3_seed1","rbt3_seed2","rbt3_seed3",
            "transformer_seed1","bilstm_seed1","rbt3_cosent","rbt3_infonce","rbt3_cls"]
    for m in models: ev("s1_"+m,"--model","s1_"+m)
    ev("s1_roberta_seed1_realidx","--model","s1_roberta_seed1","--real-index")
    ev("s1_roberta_seed1_realidx_reject","--model","s1_roberta_seed1","--real-index","--reject")
    for tag,mode,init,extra,evextra in [
        ("faqonly","faq","s1_roberta_seed1",[],[]),
        ("simcse","simcse","s1_roberta_seed1",[],[]),
        ("real","faq","s1_roberta_seed1",["--real","--real-index"],["--real-index"]),
        ("nostage1","faq","raw_roberta",final,["--real-index","--reject"])]:
        name=f"s2_{tag}_roberta_seed1"
        adapt(name,"--mode",mode,"--init",init,"--lr","1e-5",*extra,*ck)
        ev(name,"--model",name,*evextra)
    for arch,seeds,lr in [("roberta",[1,2,3],"1e-5"),("rbt3",[1,2,3],"2e-5"),
                          ("transformer",[1],"1e-4"),("bilstm",[1],"1e-4")]:
        for seed in seeds:
            suffix=f"{arch}_seed{seed}";name="s2_final_"+suffix
            adapt(name,"--mode","faq","--init","s1_"+suffix,"--lr",lr,"--seed",str(seed),*final,
                  *(ck if arch=="roberta" else []))
            ev(name,"--model",name,"--real-index","--reject","--latency")
    ev("s2_final_roberta_seed1_threshold_only","--model","s2_final_roberta_seed1","--real-index")
    mined=run(PR,"pseudo_mine",["src/adapt_domain.py","--mode","mine","--init","s2_final_roberta_seed1",
             "--real-index","--name","pseudo_seed1","--mem-frac","0.55"],
             ["results/pseudo/pseudo_seed1.json"],optional=True)
    if mined:
        ev("s2_final_roberta_seed1_pseudoidx","--model","s2_final_roberta_seed1","--real-index",
           "--reject","--extra-index","pseudo_seed1")
        adapt("s2_pseudo_roberta_seed1","--mode","faq","--init","s1_roberta_seed1","--lr","1e-5",
              *final,"--pseudo","pseudo_seed1",*ck)
        ev("s2_pseudo_roberta_seed1","--model","s2_pseudo_roberta_seed1","--real-index","--reject",
           "--extra-index","pseudo_seed1")
    run(PR,"plots",["src/plot_results.py","--tsne"])
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        project()
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
