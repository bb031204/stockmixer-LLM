"""Frozen local FinBERT inference with text-free, versioned per-article cache."""
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .io import SCHEMA, digest, digest_file, file_manifest, new_artifact, now, write_json, write_records

LABELS = ("positive", "negative", "neutral")
PREPROCESSING = "title_summary_nfkc_v1"

class LocalFinBert:
    def __init__(self, model_dir, device="cpu", max_length=512, model_available_at=None):
        directory = Path(model_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError("A local, pinned FinBERT model directory is required.")
        weights = sorted(list(directory.glob("*.safetensors"))+list(directory.glob("pytorch_model*.bin")))
        files = sorted({*weights, *directory.glob("*.json"), *directory.glob("vocab.txt")})
        if not weights:
            raise ValueError("No local model weights found.")
        if not 8 <= max_length <= 512:
            raise ValueError("max_length must be between 8 and 512.")
        hashes = {p.name:digest_file(p) for p in files}
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, __version__
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(directory),
            local_files_only=True, trust_remote_code=False).to(device).eval()
        self.model.requires_grad_(False)
        self.order = self.label_order(self.model.config.id2label)
        if self.model.config.hidden_size != 768 or self.model.config.num_labels != 3:
            raise ValueError("Expected 768-dimensional, three-label FinBERT.")
        self.max_length, self.device = max_length, device
        fingerprint = "sha256:"+digest(hashes)
        self.metadata = {"model_id":"ProsusAI/finbert", "model_revision":fingerprint,
            "tokenizer_revision":fingerprint, "local_file_sha256":hashes, "max_length":max_length,
            "embedding_method":"cls_last_hidden", "preprocessing_version":PREPROCESSING,
            "label_order":list(LABELS), "model_available_at":model_available_at,
            "transformers_version":__version__, "torch_version":str(torch.__version__),
            "python_version":platform.python_version(), "license":"cc-by-nc-sa-4.0; verify model revision"}

    @staticmethod
    def label_order(id2label):
        indices = {str(value).lower():int(key) for key,value in id2label.items()}
        if set(indices) != set(LABELS):
            raise ValueError("id2label must explicitly name positive, negative and neutral.")
        return [indices[label] for label in LABELS]

    @torch.inference_mode()
    def predict(self, articles):
        # Truncation drops the end, preserving the title prefix. No input is logged.
        texts = [(a["title"]+"\n\n"+a["summary"]).strip() for a in articles]
        tokens = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        tokens = {k:v.to(self.device) for k,v in tokens.items()}
        result = self.model(**tokens, output_hidden_states=True, return_dict=True)
        probs = torch.softmax(result.logits.float(), dim=-1)[:,self.order].cpu().numpy()
        vectors = result.hidden_states[-1][:,0].float().cpu().numpy()
        counts = tokens["attention_mask"].sum(-1).cpu().numpy()
        validate_outputs(probs,vectors,len(articles))
        return probs,vectors,counts

def validate_outputs(probs,vectors,count):
    if probs.shape != (count,3) or vectors.shape != (count,768):
        raise ValueError("Unexpected FinBERT output shape.")
    if not np.isfinite(probs).all() or not np.isfinite(vectors).all():
        raise ValueError("Nonfinite FinBERT output.")
    if np.any(probs<0) or np.any(probs>1) or not np.allclose(probs.sum(-1),1,atol=1e-5):
        raise ValueError("Invalid sentiment probabilities.")

def encode_articles(articles,backend,output,cache,batch_size=16,table_format="jsonl",input_provenance=None,progress=None):
    if Path(output).exists():
        raise FileExistsError("Output already exists; select a new version directory.")
    if not articles or batch_size<1:
        raise ValueError("Nonempty articles and positive batch_size required.")
    if table_format not in ("jsonl","parquet"):
        raise ValueError("Expected jsonl or parquet.")
    cache = Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    metadata = backend.metadata
    fingerprint = digest(metadata)
    results,pending,counts = {},[],Counter()
    for a in articles:
        if a["input_status"] != "ready":
            continue
        key = digest([a["source_id"],a["input_text_hash"],fingerprint])
        path = cache/(key+".npz")
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                p,v = saved["probs"],saved["vector"]
                validate_outputs(p[None],v[None],1)
                results[a["source_id"]]=(p,v,int(saved["token_count"]),str(saved["completed_at"]))
            counts["cache_hit"]+=1
        else:
            pending.append((a,path))
    index,size=0,batch_size
    while index<len(pending):
        chunk=pending[index:index+size]
        try:
            probs,vectors,token_counts=backend.predict([a for a,_ in chunk])
            validate_outputs(probs,vectors,len(chunk))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and size>1:
                size=max(1,size//2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                counts["oom_batch_reduction"]+=1
                continue
            for a,_ in chunk:
                results[a["source_id"]]="inference_error"
            counts["error"]+=len(chunk)
            index+=len(chunk)
            continue
        completed=now()
        for j,(a,path) in enumerate(chunk):
            vector=vectors[j].astype(np.float16)
            if not np.isfinite(vector).all():
                raise ValueError("Embedding cannot be safely stored as float16.")
            fd,temporary=tempfile.mkstemp(prefix=".cache-",suffix=".npz",dir=cache)
            os.close(fd)
            try:
                np.savez(temporary,probs=probs[j].astype(np.float32),vector=vector,
                    token_count=token_counts[j],completed_at=np.array(completed))
                os.replace(temporary,path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            results[a["source_id"]]=(probs[j],vector,int(token_counts[j]),completed)
            counts["inferred"]+=1
        index+=len(chunk)
        if progress is not None:
            progress(index,len(pending),dict(counts))
    rows,vectors=[],[]
    for a in articles:
        row={k:v for k,v in a.items() if k not in ("title","summary","input_status")}
        row.update(model_revision=metadata["model_revision"],model_id=metadata["model_id"],
            tokenizer_revision=metadata["tokenizer_revision"],schema_version=SCHEMA,
            model_available_at=metadata.get("model_available_at"),embedding_row=None)
        result=results.get(a["source_id"])
        if isinstance(result,tuple):
            p,v,token_count,completed=result
            row.update(finbert_status="success",p_positive=float(p[0]),p_negative=float(p[1]),
                p_neutral=float(p[2]),token_count=token_count,embedding_row=len(vectors),embedding_dim=768,
                finbert_completed_at=completed,news_feature_available_at=completed)
            vectors.append(v)
        else:
            row["finbert_status"]=result if isinstance(result,str) else a["input_status"]
        rows.append(row)
    with new_artifact(output) as directory:
        write_records(directory/("article_finbert."+table_format),rows)
        np.save(directory/"article_embeddings_fp16.npy",np.asarray(vectors,dtype=np.float16).reshape(-1,768))
        manifest={"schema_version":SCHEMA,"kind":"article_features","created_at":now(),
            "encoder":metadata,"encoder_fingerprint":fingerprint,"articles":len(rows),"successful":len(vectors),
            "counts":dict(counts),"status_counts":dict(Counter(r["finbert_status"] for r in rows)),
            "record_table":"article_finbert."+table_format,
            "source_input_hash":digest([[a["source_id"],a["input_text_hash"]] for a in articles]),
            "input_provenance":input_provenance,
            "files":file_manifest(directory)}
        write_json(directory/"manifest.json",manifest)
    return manifest
