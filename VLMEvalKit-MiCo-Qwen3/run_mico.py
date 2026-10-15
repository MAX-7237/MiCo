#!/usr/bin/env python3
"""MiCo launcher: maps model names onto MiCo configurations."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
profiles=json.loads((root/'mico_models.json').read_text())
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True,choices=profiles)
p.add_argument('--tokens',required=True,type=int,help='Exact layer-average T; image NeXT uses e.g. 640,320,160; video uses per-frame budget.')
p.add_argument('--variant',choices=['mico','mico-s1'],default='mico')
p.add_argument('--dataset',help='VLMEval or lmms_eval task name')
p.add_argument('--model-path')
p.add_argument('--question-file')
p.add_argument('--image-folder')
p.add_argument('--answers-file')
p.add_argument('--conv-mode',default='vicuna_v1')
p.add_argument('--work-dir',default='outputs/mico')
p.add_argument('--nproc',type=int,default=1)
p.add_argument('--port',type=int,default=29531)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
if a.tokens<=0 or a.nproc<=0:p.error('tokens and nproc must be positive')
m=profiles[a.model];method=m['method']
if a.variant=='mico-s1':
 if m['kind']=='video':p.error('Video S1-only alias has not been verified')
 method='mico_s1'
env=os.environ.copy()
paths=[str(root)]
if m['kind']=='video':paths.insert(0,str(root/'lmms-eval'))
env['PYTHONPATH']=os.pathsep.join(paths+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
if m['kind']=='vlmeval':
 if not a.dataset:p.error('--dataset is required')
 if a.model_path:p.error('For VLMEval, edit the model registry in vlmeval/config.py to change weights')
 cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node='+str(a.nproc),'--master_port='+str(a.port),'run.py','--model',m['registry'],'--data',a.dataset,'--pruning-method',method,'--visual-token-num',str(a.tokens),'--work-dir',a.work_dir]
elif m['kind']=='image':
 if not all([a.question_file,a.image_folder,a.answers_file]):p.error('--question-file, --image-folder and --answers-file are required')
 if a.nproc!=1:p.error('Image entry uses one process; use original chunk scripts for distributed evaluation')
 cmd=[sys.executable,'-m','llava.eval.model_vqa_loader','--model-path',a.model_path or m['model_path'],'--question-file',a.question_file,'--image-folder',a.image_folder,'--answers-file',a.answers_file,'--pruning_method',method,'--visual_token_num',str(a.tokens),'--temperature','0','--conv-mode',a.conv_mode]
else:
 if not a.dataset:p.error('--dataset is required')
 model_args='pretrained='+ (a.model_path or m['model_path'])+',conv_template=qwen_1_5,max_frames_num=64,pruning_method='+method+',visual_token_num='+str(a.tokens)
 cmd=[sys.executable,'-m','accelerate.commands.launch','--num_processes',str(a.nproc),'--main_process_port',str(a.port),'-m','lmms_eval','--model','llava_vid','--model_args',model_args,'--tasks',a.dataset,'--batch_size','1','--log_samples','--output_path',a.work_dir]
print('MiCo method:',method,flush=True)
print('Working directory:',root,flush=True)
print(shlex.join(cmd),flush=True)
if not a.dry_run:
 if a.answers_file:
  answer=Path(a.answers_file).expanduser()
  if not answer.is_absolute():answer=root/answer
  answer.parent.mkdir(parents=True,exist_ok=True)
 raise SystemExit(subprocess.call(cmd,cwd=root,env=env))
