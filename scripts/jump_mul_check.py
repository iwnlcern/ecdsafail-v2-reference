"""Exact enabled Jump-4 composition, fresh inverse, boundary and peak checks.

Credit ab-godder's arith.add with a one-CCX top bit (PR #9, PR #28).
No public x=0 promise. Expected states are independent modular integers.
"""
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from qec import model,sim
from qec.arith import Mod
from qec.builder import Circuit,CCX
from qec.point_add import DialogMultiplier,ControlledPointAdd
from qec.curve import SECP256K1,SECP256K1_G
from jump_step_check import ReleaseCircuit,release_run,bad_wires,split_flat,primes


def build(q,fault=None,public=False):
    C=ReleaseCircuit();m=DialogMultiplier(C,Mod(q));n=q.bit_length()
    original=C.apply
    boundaries=[]
    def apply(block,regs,inverse=False):
        if len(regs)==4 and [len(r) for r in regs]==[n,n+m._jump4_pool_sizes()[2],m._jump4_junk_size(),3]:
            boundaries.append(dict(inverse=inverse,block_id=id(block),at=len(C.prims),regs=[list(r) for r in regs]))
            if inverse:
                if fault in ('dirty_E','dirty_H','dirty_G'):
                    C.x(regs[1][m._jump4_pool_sizes()[1]] if fault=='dirty_G' else regs[{'dirty_E':2,'dirty_H':3}[fault]][0])
                if fault=='wrong_fresh_alias': regs[1][0]=regs[0][0]
        original(block,regs,inverse)
    C.apply=apply
    if fault=='lost_top':
        shared=m._shared_difference
        def lose(U,V,*args,**kwargs):
            at=len(C.prims);shared(U,V,*args,**kwargs)
            pos=max(i for i in range(at,len(C.prims),4) if C.prims[i]==1 and C.prims[i+2]==V[-1])
            # Omit the actual final high-bit write, updating observed offsets.
            del C.prims[pos:pos+4]
            C.events=[(p-4 if p>pos else p,k,w) for p,k,w in C.events]
        m._shared_difference=lose
    if fault=='premature_zero_lending':
        base_apply=C.apply;fired=False
        def dirty_loan(block,regs,inverse=False):
            nonlocal fired
            if not fired and len(regs) in (5,9) and len(regs[2])==1 and len(regs[3])==4:
                # Fault the real live top U before its first subtraction: an
                # early clean loan frees/reallocates and writes that nonzero
                # wire. A harmless altered D in late padding is NOT a control.
                w=regs[0][-1];C.free([w]);assert C.alloc(1)==[w];C.x(w);fired=True
            base_apply(block,regs,inverse)
        C.apply=dirty_loan
    block=C.record_shape((lambda x,y:m.apply(x,y)) if public else m._jump4_body,[n,n])
    assert len(boundaries)==2 and not boundaries[0]['inverse'] and boundaries[1]['inverse']
    assert boundaries[0]['block_id']==boundaries[1]['block_id'], 'different Euclid replay block'
    return block,boundaries,m


def check(q,cases,fault=None):
    block,bounds,m=build(q,fault)
    public_block=build(q,public=True)[0] if fault is None else None
    n=q.bit_length();ins=split_flat(block.flat_in,[n,n]);outs=split_flat(block.flat_out,[n,n])
    # Different physical layout and live spectators; inverse from expected
    # external values, never the forward simulator's returned state.
    D=Circuit();spectator=D.alloc(n+7);regs=[D.alloc(n),D.alloc(n)];entry=[list(r) for r in regs]
    D.apply(block,regs,inverse=True)
    forward_bad=inverse_bad=release_bad=0
    for at in range(0,len(cases),2048):
        chunk=cases[at:at+2048];xs,ys=zip(*chunk);zs=[x*y%q for x,y in chunk]
        initial=sim.pack(xs,ins[0])|sim.pack(ys,ins[1]);want=sim.pack(xs,outs[0])|sim.pack(zs,outs[1])
        if fault:
            actual=sim.run(block.ops,max(block.wires|set(block.flat_in)|set(block.flat_out))+1,initial,len(chunk))
            forward_bad+=bad_wires(actual,want)
            try:release_run(block,initial,len(chunk))
            except AssertionError:release_bad+=1
        else:
            assert bad_wires(release_run(block,initial,len(chunk)),want)==0,('mul forward',q,at)
            assert bad_wires(release_run(block,want,len(chunk),True),initial)==0,('mul inverse/release',q,at)
            pi,po = split_flat(public_block.flat_in,[n,n]),split_flat(public_block.flat_out,[n,n])
            public_in=sim.pack(xs,pi[0])|sim.pack(ys,pi[1])
            public_out=sim.pack(xs,po[0])|sim.pack(zs,po[1])
            assert bad_wires(release_run(public_block,public_in,len(chunk)),public_out)==0,('public apply forward',q,at)
            assert bad_wires(release_run(public_block,public_out,len(chunk),True),public_in)==0,('public apply inverse',q,at)
        si=sim.pack(xs,entry[0])|sim.pack(zs,entry[1])|sim.pack([((x<<3)^y)%(1<<(n+7)) for x,y in chunk],spectator)
        sw=sim.pack(xs,regs[0])|sim.pack(ys,regs[1])|sim.pack([((x<<3)^y)%(1<<(n+7)) for x,y in chunk],spectator)
        ib=bad_wires(sim.run(D.prims,D.num_wires,si,len(chunk)),sw)
        if fault:inverse_bad+=ib
        else: assert ib==0,('fresh inverse/spectator',q,at)
    if fault:
        assert forward_bad and inverse_bad,('uncaught composition control',fault,forward_bad,inverse_bad,release_bad)
        return dict(cases=len(cases),forward_bad=forward_bad,inverse_bad=inverse_bad,dirty_release_batches=release_bad)
    return dict(q=q,cases=len(cases),forward_bad=0,inverse_bad=0,releases=len(block.events),
                release_scope='observed composition events; nested cells/producer checked separately',
                ccx=block.ops[::4].count(CCX),wires=len(set(block.flat_in)|block.wires),
                keep=0,copies=0,same_recorded_block=True,fresh_inverse=True,public_apply=True)


def count_graph(ops,calls,memo=None):
    memo={} if memo is None else memo
    count=ops[::4].count(CCX)
    for call in calls or ():
        key=id(call.block)
        if key not in memo: memo[key]=count_graph(call.block.ops,call.block.calls,memo)
        count+=memo[key]
    return count


def resources():
    gen=ControlledPointAdd(SECP256K1,SECP256K1_G)
    phase_allocations=[];public=DialogMultiplier.apply
    def traced(m,X,Y,inverse=False,test=None,flag=None):
        active=m.C.num_wires-len(m.C.free_stack)
        row=dict(inverse=inverse,operand_wires=len(X)+len(Y),outer_live=active-len(X)-len(Y),active_before=active)
        result=public(m,X,Y,inverse,test,flag)
        row['construction_peak_after']=m.C.num_wires
        phase_allocations.append(row)
        return result
    with patch.object(DialogMultiplier,'apply',traced):C=gen.build(True)
    ccx=count_graph(C.prims,C.calls)
    # Independent component construction, counted with the same primitive/call
    # expansion, reconciles the whole without priced/unbuilt credits.
    D=Circuit(True);m=DialogMultiplier(D,Mod(model.SECP256K1_Q));n=m.mod.n;T=m.iters
    producer=D.record_shape(m._jump4_euclid_pass,[n,n+m._jump4_pool_sizes()[2],m._jump4_junk_size(),3])
    coefficient=D.record_shape(m._jump4_bezout_pass,[4*T,3,n,n])
    ep=count_graph(producer.ops,producer.calls);bp=count_graph(coefficient.ops,coefficient.calls)
    sg=ControlledPointAdd(SECP256K1,SECP256K1_G);Q=Circuit(True)
    sg.C=Q;sg._modadd_blk=sg._square_blk=None
    g,x,y=Q.alloc(1),Q.alloc(n),Q.alloc(n)
    sg._mulsub(g[0],y,x)
    square=count_graph(Q.prims,Q.calls)
    parts=dict(euclid_each=ep,euclid_traversals=4,coefficient_each=bp,coefficient_traversals=2,
               square=square,glue=ccx-4*ep-2*bp-square)
    assert parts['glue']>=0
    assert C.num_wires<=4096,('whole point-add wires',C.num_wires)
    # Raw count is a construction census only; the scored artifact and full gates are separate checks and cannot be inferred from this component test.
    phase_shapes=dict(producer_inputs=len(producer.flat_in),
                      producer_internal=len(producer.wires-set(producer.flat_in)),
                      coefficient_inputs=len(coefficient.flat_in),
                      coefficient_internal=len(coefficient.wires-set(coefficient.flat_in)))
    return dict(raw_ccx=ccx,raw_T=ccx*7,wires=C.num_wires,mode='hierarchical construction',scored=None,
                components=parts,multiplier_phase_allocations=phase_allocations,phase_shapes=phase_shapes)


def flags_and_shapes():
    code='import json;from qec import model;print(json.dumps(dict(flag=model.JUMP4)))'
    rows={}
    for value in (None,'0','1','invalid'):
        env=dict(os.environ)
        if value is None: env.pop('QEC_JUMP4',None)
        else: env['QEC_JUMP4']=value
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
        if value=='invalid': assert p.returncode!=0
        else: assert p.returncode==0 and json.loads(p.stdout)['flag']==(value=='1')
        rows['unset' if value is None else value]=dict(exit=p.returncode)
    C=Circuit();m=DialogMultiplier(C,Mod(31));x,y=C.alloc(5),C.alloc(5)
    for a,b in ((x,x),(x[:-1],y)):
        try:m.apply(a,b)
        except ValueError:pass
        else:raise AssertionError('malformed/aliased enabled operands accepted')
    assert m.split(x,y) is False and m._halves is None
    for kwargs in ({'test':lambda *args:None},{'flag':x[0]}):
        try:m.apply(x,y,**kwargs)
        except ValueError:pass
        else:raise AssertionError('unsupported legacy cut input accepted')
    old=model.JUMP4
    try:
        model.JUMP4=not old
        try:m.apply(x,y)
        except ValueError:pass
        else:raise AssertionError('instance construction mode changed')
    finally:model.JUMP4=old
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['exact'],required=True)
    ap.add_argument('--max-q',type=int,default=127);ap.add_argument('--secp-directed',action='store_true')
    ap.add_argument('--controls',action='store_true');ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();assert os.environ.get('QEC_JUMP4')=='1' and model.JUMP4,'startup flag not enabled'
    assert hasattr(DialogMultiplier,'_jump4_body'),'composed multiplier missing'
    rows=[]
    for q in primes(args.max_q):
        rows.append(check(q,[(x,y) for x in range(1,q) for y in range(q)]))
        print(json.dumps(dict(q=q,status='pass',cases=rows[-1]['cases'])),flush=True)
    if args.secp_directed:
        q=model.SECP256K1_Q;rng=random.Random(7979)
        xs=sorted({z for k in range(256) for base in (1<<k,q-(1<<k)) for z in (base-1,base,base+1) if 0<z<q})
        cases=[(x,y) for x in xs for y in (0,1,q-1)]+[(rng.randrange(1,q),rng.randrange(q)) for _ in range(512)]
        rows.append(check(q,cases))
    controls={}
    if args.controls:
        for fault in ('dirty_E','dirty_H','dirty_G','wrong_fresh_alias','lost_top','premature_zero_lending'):
            controls[fault]=check(127,[(x,y) for x in range(1,127) for y in (0,1,2,126)],fault)
        assert controls['premature_zero_lending']['dirty_release_batches']>0
    result=dict(status='pass',rows=rows,controls=controls,resources=resources(),startup=flags_and_shapes(),flag=1,
                command=sys.argv,head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                source={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                        ('qec/model.py','qec/point_add.py','qec/two_bit.py','scripts/jump_mul_check.py','scripts/jump_step_check.py')})
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'mul.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
