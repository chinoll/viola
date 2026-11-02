"""Warmed full Linear forward/backward, including grouped quantization/layouts."""

import argparse
import gc
import json
from pathlib import Path
from statistics import median
import time

import torch

import viola
from viola.nn import _grouped_kernels as grouped
from viola.nn._grouped_tuning import get_config


def timing(step):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(4):
                step()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):
            graph.replay()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) / 32)
    return dict(median_ms=median(samples), samples_ms=samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shape', default='4096,4096,4096', help='tokens,input,output')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--strict-bf16-reduction', action='store_true')
    args = p.parse_args()
    m, k, n = map(int, args.shape.split(','))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not args.strict_bf16_reduction
    viola.set_master_weight_dtype('fp32'); viola.set_gradient_dtype('bf16')
    report = dict(scope=__doc__, gpu=torch.cuda.get_device_name(), shape=[m,k,n],
                  no_optimizer_math=True, mode='memory', rows=[], complete=False,
                  bf16_reduced_precision_reduction=not args.strict_bf16_reduction)
    reference = None
    for scaling in ('bf16','grouped','mxfp8'):
        torch.manual_seed(77)
        source = torch.nn.Linear(k,n,device='cuda',dtype=torch.bfloat16)
        model = source if scaling == 'bf16' else viola.nn.Linear.from_float(source,scaling=scaling)
        del source
        opt = viola.optim.bind_optimizer(model,torch.optim.SGD(model.parameters(),lr=.001))
        # from_float initializes a temporary Linear before copying its weights.
        # Reset the input RNG so all backends receive identical X and dY.
        torch.manual_seed(177)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
        def clear():
            opt.zero_grad(); x.grad=None
        def step():
            clear()
            y=model(x); y.backward(dy)
            return y
        y=step()
        if scaling=='bf16':
            reference={key:value.detach().cpu() for key,value in dict(y=y,dx=x.grad,dw=model.weight.grad).items()}
        del y
        row=dict(scaling=scaling,default=timing(step))
        print(json.dumps(dict(event='default',**row)),flush=True)
        if scaling!='bf16':
            model.autotune=True
            before=time.monotonic(); step(); torch.cuda.synchronize()
            row['first_search_seconds']=time.monotonic()-before
            row['autotuned']=timing(step)
            # Exercise persisted configurations after dropping the Python lookup cache.
            get_config.cache_clear()
            before=time.monotonic(); step(); torch.cuda.synchronize()
            row['disk_cache_reload_seconds']=time.monotonic()-before
        clear()
        gc.collect(); torch.cuda.empty_cache()
        before=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        y=step(); torch.cuda.synchronize()
        tensors=[x,dy,*model.parameters(),*model.buffers(),*(p for group in opt.param_groups for p in group['params'])]
        owned={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
        row['persistent_with_inputs_mib']=sum(owned.values())/2**20
        row['before_step_allocated_mib']=before/2**20
        row['step_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        row['step_extra_peak_mib']=(torch.cuda.max_memory_allocated()-before)/2**20
        if scaling!='bf16':
            actual=dict(y=y,dx=x.grad,dw=model.weight.grad)
            row['versus_bf16']={key:((v.detach().float().cpu()-reference[key].float()).norm()/reference[key].float().norm()).item() for key,v in actual.items()}
            del actual
        del tensors
        report['rows'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(event='complete_case',**row)),flush=True)
        clear(); del step,clear,opt,model,x,dy,y
        gc.collect(); torch.cuda.empty_cache()
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
