"""Package-local frozen-output reproduction. Never train, infer or fit a threshold."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics

from frozen_metrics import ap,auc,binary,calibration,legacy_constructed_ap,prf,quantile,require,sha256,unique_index

ROOT=Path(__file__).resolve().parents[1]


def read(relative):
    p=(ROOT/relative).resolve()
    require(ROOT in p.parents,'Input outside candidate package')
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def check_hashes():
    with (ROOT/'core_release_sha256.tsv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    unique_index(rows,'path')
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    expected={r['path'] for r in rows}
    require(actual==expected|{'core_release_sha256.tsv'},'Hash manifest does not cover exact core tree')
    for r in rows:
        p=(ROOT/r['path']).resolve()
        require(ROOT in p.parents and sha256(p)==r['sha256'],'Core checksum mismatch')
    return dict(files=len(actual),verified=len(rows))


def human300():
    gold=read('annotations/human300_final_labels.csv');scores=read('scores/human300_three_model_scores.csv')
    g=unique_index(gold,'sample_id');s=unique_index(scores,'sample_id')
    require(set(g)==set(s) and len(g)==300,'Human300 stable IDs mismatch')
    require(Counter(r['final_label_0_1_2_U'] for r in gold)=={'0':62,'1':92,'2':146},'Human300 label counts')
    protocol=json.loads((ROOT/'configs/human_annotation_protocol.json').read_text(encoding='utf-8'))
    require(protocol['pre_adjudication_agreement']==254 and protocol['manually_adjudicated_disagreements']==46,'Author-confirmed process metadata mismatch')
    test=set(unique_index(read('splits/test_ids.csv'),'sample_id'))
    require(set(g)<=test,'Human300 ID not in original project test split')
    out=[]
    archive=read('tables/table6_human300_all_metrics.csv')
    for model,prefix in [('Semantic-HN','semantic_hn'),('FT-CLIP','ft_clip'),('Random-HN','random_hn')]:
        for task,positive in [('strict_support',{'2'}),('broad_relatedness',{'1','2'})]:
            y=[];q=[];pred=[]
            for key,row in s.items():
                require(row['final_label_0_1_2_U']==g[key]['final_label_0_1_2_U'],'Frozen Gold mismatch')
                y.append(int(row['final_label_0_1_2_U'] in positive));q.append(float(row[prefix+'_probability']))
                pred.append(int(row[prefix+'_prediction']))
                require(pred[-1]==int(q[-1]>=float(row[prefix+'_threshold'])),'Frozen threshold/prediction mismatch')
            result=dict(model=model,task=task,threshold=float(scores[0][prefix+'_threshold']),**binary(y,q,pred),**calibration(y,q))
            match=[r for r in archive if r['model']==model and r['task']==task]
            require(len(match)==1,'Missing archived human300 row')
            for key,field in [('auroc','auc_probability'),('ap','ap_probability'),('f1','f1'),('specificity','specificity'),('brier_score','brier_score'),('ece','ece')]:
                require(abs(result[key]-float(match[0][field]))<1e-12,'Human300 metric differs from frozen archive: '+key)
            out.append(result)
    require(round(out[0]['auroc'],4)==.5842,'Strict Semantic-HN AUROC mismatch')
    return dict(mode='frozen-output reproduction',n=300,label_counts={'0':62,'1':92,'2':146},
        reported_pre_adjudication_agreement=protocol['pre_adjudication_agreement'],
        reported_manually_adjudicated=protocol['manually_adjudicated_disagreements'],
        process_counts_source='author-confirmed aggregate metadata, not recomputed from private per-annotator labels',metrics=out)


def relation200():
    gold=read('annotations/historical200_gold_relation_counts.csv')
    unique_index(gold,('blind_id','relation_type'))
    require(len({r['blind_id'] for r in gold})==200,'Gold sample count')
    gtype=Counter()
    for r in gold:gtype[r['relation_type']]+=int(r['gold_relation_count'])
    require(gtype=={'OCCURS_IN':64,'AFFECTS':8,'CAUSES':2},'Historical relation Gold count')
    flags=read('predictions/relation200_match_flags.csv')
    unique_index(flags,('protocol','stage','blind_id','within_sample_relation_index'))
    counts=defaultdict(Counter)
    for r in flags:
        key=(r['protocol'],r['stage'],r['blind_id'],r['relation_type'])
        counts[key]['tp' if int(r['strict_gold_match']) else 'fp']+=1
    archived=read('metrics/relation200_per_sample_strict_counts.csv')
    unique_index(archived,('protocol','stage','blind_id','relation_type'))
    groups={ (r['protocol'],r['stage']) for r in archived }
    out=[]
    for protocol,stage in sorted(groups):
        total=Counter(tp=0,fp=0,fn=0)
        for g in gold:
            key=(protocol,stage,g['blind_id'],g['relation_type'])
            tp=counts[key]['tp'];fp=counts[key]['fp'];fn=int(g['gold_relation_count'])-tp
            require(fn>=0,'Impossible TP > Gold')
            source=[r for r in archived if (r['protocol'],r['stage'],r['blind_id'],r['relation_type'])==key]
            require(len(source)==1 and tuple(int(source[0][k]) for k in ('tp','fp','fn'))==(tp,fp,fn),'Derived match flags differ from frozen per-sample counts')
            total.update(tp=tp,fp=fp,fn=fn)
        out.append(dict(protocol=protocol,stage=stage,relation_count=total['tp']+total['fp'],**prf(**total)))
    expected={('historical200','candidate'):(50,86,24),('historical200','reliable'):(17,14,57),
      ('historical200','reliable_review'):(48,60,26),('prospective_reused200','candidate'):(57,151,17),
      ('prospective_reused200','reliable'):(18,29,56),('prospective_reused200','reliable_review'):(49,107,25)}
    for r in out:require(tuple(r[k] for k in ('tp','fp','fn'))==expected[(r['protocol'],r['stage'])],'Historical/prospective frozen metrics mismatch')
    return dict(mode='frozen-output reproduction from precomputed strict-match flags; endpoints withheld',gold_relations=74,gold_types=dict(gtype),metrics=out)


def fixed250():
    rows=read('scores/fixed250_nli_scores_labels.csv');unique_index(rows,'item_id')
    require(len(rows)==250 and Counter(r['label'] for r in rows)=={'1':218,'0':32},'Fixed250 Gold mismatch')
    out=[]
    for stage,allowed in [('all',{'reliable','review','reject'}),('reliable',{'reliable'}),('reliable_review',{'reliable','review'})]:
        y=[int(r['label']) for r in rows];q=[float(r['nli_only_score']) for r in rows]
        pred=[int(r['nli_only_tier'] in allowed) for r in rows]
        out.append(dict(stage=stage,**binary(y,q,pred)))
    require(out[1]['tp']==99 and out[1]['fp']==3 and round(out[1]['precision'],4)==.9706,'Fixed250 Reliable mismatch')
    return dict(mode='frozen-output reproduction',n=250,supported=218,unsupported=32,metrics=out)


def pair_table(table):
    config=json.loads((ROOT/'configs/pair_replays.json').read_text(encoding='utf-8'))[table]
    results=[]
    for item in config:
        rows=read(item['file'])
        if table=='table4':unique_index(rows,'pair_id')
        for subset in ('overall','hard'):
            selected=rows if subset=='overall' else [r for r in rows if r['kind'] in {'positive','semantic_hard_negative','label_matched_hard_negative'}]
            y=[int(r['label']) for r in selected];s=[float(r['score']) for r in selected];pred=[int(r['prediction']) for r in selected]
            values=binary(y,s,pred)
            if table=='table4':values['ap']=legacy_constructed_ap(y,s)
            results.append(dict(model=item['model'],subset=subset,ap_convention='historical stable-rank' if table=='table4' else 'grouped-score AP',**values))
    archive=read('tables/'+table+'_comparison.csv')
    aliases={'Fine-tuned CLIP':'FT-CLIP','Random-HN CLIP':'Random-HN','Semantic-HN CLIP':'Semantic-HN'}
    for result in results:
        if table=='table4':
            for metric,field in [('AUC','auroc'),('AP','ap'),('F1','f1')]:
                row=next(r for r in archive if r['model']==aliases.get(result['model'],result['model']) and r['subset']==('semantic_hard' if result['subset']=='hard' else 'overall') and r['metric']==metric)
                require(abs(result[field]-float(row['score']))<1e-7,'Constructed-pair replay differs from archived main cell')
        else:
            row=next(r for r in archive if r['model']==result['model'])
            fields=[('auroc','auc'),('ap','average_precision'),('f1','f1')] if result['subset']=='overall' else [('auroc','hard_auc')]
            for field,key in fields:require(abs(result[field]-float(row[key]))<1e-10,'MASH frozen score replay differs from archived metric')
    return dict(mode='frozen-output reproduction using archived scores and predictions; no new thresholds',metrics=results,
        archived_paper_cells=archive)


def verifier_table8():
    scores=read('scores/fixed250_nli_scores_labels.csv');single=read('scores/fixed250_single_template_scores.csv')
    s=unique_index(scores,'item_id');ss=unique_index(single,'item_id')
    require(set(s)==set(ss) and len(s)==250,'Single/multi-template IDs mismatch')
    archived=read('tables/table8_negative_sensitive_metrics.csv');out=[]
    for row in archived:
        y=[int(r['label']) for r in scores]
        if row['method']=='All-positive baseline':q=[1.]*len(y)
        elif row['method']=='Single-template NLI':q=[float(ss[(r['item_id'],)]['single_template_nli_score']) for r in scores]
        else:q=[float(r['q_nli']) for r in scores]
        threshold=float(row['threshold']);pred=[int(v>=threshold) for v in q]
        result=dict(method=row['method'],threshold=threshold,**binary(y,q,pred))
        for key,source in [('auroc','auc'),('ap','ap'),('precision','precision'),('recall','recall'),('specificity','specificity'),('balanced_accuracy','balanced_accuracy'),('mcc','mcc'),('negative_f1','negative_f1')]:
            require(abs(result[key]-float(row[source]))<1e-12,'Table8 differs from frozen value: '+key)
        out.append(result)
    return dict(mode='frozen-output reproduction of fixed-candidate raw NLI; unchanged archived thresholds',metrics=out)


def archive_table(table):
    config=json.loads((ROOT/'configs/table_replays.json').read_text(encoding='utf-8'))[table]
    out=[]
    for relative in config['files']:
        if relative.endswith('.json'):
            payload=json.loads((ROOT/relative).read_text(encoding='utf-8'))
            out.append(dict(file=relative,descriptive_metadata=payload))
            continue
        rows=read(relative)
        # Where original aggregated TP/FP/FN exist, check PRF without reconstructing unavailable endpoints.
        checked=0
        for row in rows:
            if all(k in row for k in ('tp','fp','fn','precision','recall','f1')):
                values=prf(int(row['tp']),int(row['fp']),int(row['fn']))
                require(all(abs(float(row[k])-values[k])<1e-10 for k in ('precision','recall','f1')),'Archived count/metric mismatch')
                checked+=1
        out.append(dict(file=relative,rows=rows,aggregated_prf_rows_verified=checked))
    return dict(mode=config['mode'],limitation=config['limitation'],sources=out)


def deployment():
    results=archive_table('table10');rows=read('metrics/query_latency_records.csv')
    require(len(rows)==1134,'Query latency count mismatch')
    measured=[]
    for relation in sorted({r['relation'] for r in rows}):
        values=[float(r['latency_ms']) for r in rows if r['relation']==relation]
        measured.append(dict(relation=relation,queries=len(values),mean_ms=statistics.mean(values),median_ms=statistics.median(values),p95_ms=quantile(values,.95)))
    items=read('metrics/deployment_summary.csv');index=unique_index(items,'metric_path')
    values={k[0]:float(r['value']) for k,r in index.items() if r['value'] not in {'True','False'}}
    total=values['workload.total_candidates'];review=values['workload.manual_review_candidates']
    require((total,review,values['workload.automatic_write_candidates'])==(6721,3587,2550),'Deployed candidate counts')
    reduction=1-review/total;require(round(reduction*100,2)==46.63,'Queue-count reduction')
    results.update(recomputed_latency=measured,review_queue_candidate_count_reduction=reduction,
        not_human_time_reduction=True,not_live_database_replay=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    choices=['all','core','table1','table2','table3','table4','table5','table6','table7','table8','table9','table10','table11','tableA1','fig4','fig5','fig6','fig7','fig8','fig9','fig10']
    p.add_argument('--target',choices=choices,default='core');args=p.parse_args()
    hashes=check_hashes();out={'scope':'frozen-output reproduction, not full end-to-end reproduction','hash_check':hashes}
    if args.target in {'core','all','table6'}:out['human300']=human300()
    if args.target in {'core','all','table11','fig6'}:out['historical_and_prospective200']=relation200()
    if args.target in {'core','all','table8','table9','fig7','fig8'}:out['fixed250']=fixed250()
    targets=[]
    if args.target=='all':targets=['table1','table2','table3','table4','table5','table7','table8','table9','table10','table11','tableA1','fig5','fig6','fig7','fig8','fig9']
    elif args.target not in {'core','table6'}:targets=[{'fig4':'table4','fig10':'table10'}.get(args.target,args.target)]
    for target in targets:
        if target in {'table4','table5'}:out[target]=pair_table(target)
        elif target=='table8':out[target]=verifier_table8()
        elif target=='table10':out[target]=deployment()
        else:out[target]=archive_table(target)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
