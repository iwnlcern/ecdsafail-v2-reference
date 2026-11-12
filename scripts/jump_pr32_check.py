"""PR32 Jump4 actual shrinking-pool and whole integration checks.

Built on jtaroreh/lowt-climb PR #32, carrying our PR #2 shrinking pool.
Credit ab-godder's arith.add with a one-CCX top bit (PR #9, PR #28).
No peer executable imports or retained large emission copies.
"""
import argparse, hashlib, io, json, os, random, subprocess, sys
from pathlib import Path
from array import array
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from qec import model,sim
from qec.arith import Mod
from qec.builder import Circuit,CCX
from qec.point_add import DialogMultiplier,ControlledPointAdd
from qec.curve import SECP256K1,SECP256K1_G,toy_secp_curves,Curve
from jump_step_check import ReleaseCircuit,release_run,bad_wires,split_flat,reference_cell,expected,primes,secp_inputs


def check_pool(q,cp=0,xs=None,inverse=True,controls=False,carry_limit=None,high_min=None,through_B=False,fault=None):
    if fault=='wrong_K_minus2':
        from unittest.mock import patch
        from qec import point_add
        true=point_add.tick_joint
        with patch.object(point_add,'tick_joint',lambda *a:[max(1,k-2) for k in true(*a)]):
            return check_pool(q,cp,xs,inverse,controls,carry_limit,high_min,through_B,'wrong_K_active')
    xs=list(range(1,q)) if xs is None else list(xs)
    C=ReleaseCircuit();m=DialogMultiplier(C,Mod(q));n,T=m.mod.n,m.iters
    if carry_limit is not None:m.jump4_carry_limit=carry_limit
    if high_min is not None:m.jump4_high_min=high_min
    plan=m._jump4_eager_plan();junk_n=m._jump4_junk_size()
    fired=['K-2'] if fault=='wrong_K_active' else []
    future=set()
    if fault=='b_for_g':
        original_update=m._jump4_update_flag
        def wrong(U,g,b,flag,slot,zg,k,mode):
            # zg was computed with g before the difference. Replace that
            # product and the update control, including the q97/x65 drop.
            C.ccx(g,flag[0],zg);C.ccx(b,flag[0],zg)
            original_update(U,b,b,flag,slot,zg,k,mode);fired.append(mode)
        m._jump4_update_flag=wrong
    elif fault=='skipped_flag_drop':
        original_update=m._jump4_update_flag
        def skip(U,g,b,flag,slot,zg,k,mode):
            if mode=='drop':fired.append(mode)
            else:original_update(U,g,b,flag,slot,zg,k,mode)
        m._jump4_update_flag=skip
    elif fault in ('dirty_carry','dirty_flag'):
        original_difference=m._jump4_difference
        def dirty(U,V,g,b,**kw):
            original_difference(U,V,g,b,**kw)
            wires=kw.get('junk') if fault=='dirty_carry' else ([kw['held'][0]] if kw.get('held') else [])
            if wires and not fired:C.x(wires[0]);fired.append(fault)
        m._jump4_difference=dirty
    elif fault=='high_bit_omission':
        original_high=m._high_word
        def omit(U,V,*args,**kw):
            start=len(C.prims);original_high(U,V,*args,**kw)
            if not fired:
                pos=max(i for i in range(start,len(C.prims),4) if C.prims[i]==1 and C.prims[i+2]==V[-1])
                del C.prims[pos:pos+4];C.events=[(p-4 if p>pos else p,k,w) for p,k,w in C.events];fired.append(fault)
        m._high_word=omit

    u_lent,lent,extra=m._jump4_pool_sizes()
    sizes=[n,n+extra,junk_n,3,cp]+([n] if through_B else [])
    loans=[];apply=C.apply;t=0
    def observe(block,regs,inverse=False):
        nonlocal t
        if len(regs) in (5,9) and len(regs[2])==1 and len(regs[3])==4:
            future.difference_update(regs[3])
            if fault=='early_future_tape' and len(regs)==9 and not fired:
                candidates=[w for w in regs[8] if w in future]
                if candidates:
                    w=candidates[0];C.free([w]);assert C.alloc(1)==[w];C.x(w);fired.append(w)
        apply(block,regs,inverse)
        if len(regs) in (5,9) and len(regs[2])==1 and len(regs[3])==4:
            nw=m.widths[t+1] if t+1<T else m.pass_tail_width
            tails=list(regs[0][nw:])+list(regs[1][nw:])
            loans.append((len(C.prims),tails));t+=1
            # Explicit boundary assertions, not allocator releases. Check both
            # directions at the exact moment before any future tape reuse.
            C.events.extend([(len(C.prims),'free',tails),(len(C.prims),'alloc',tails)])
    C.apply=observe
    def body(U,pool,J,H,copies,*coefficient):
        future.update(pool[n:])
        m._jump4_euclid_pass(U,pool,J,H)
        if coefficient:
            C.x(U[0]);G=m._jump4_coded_tape(pool)
            m._jump4_bezout_pass(G,H,coefficient[0],U);pool[lent:]=G
    block=C.record_shape(body,sizes)
    assert t==T
    ins=split_flat(block.flat_in,sizes);outs=split_flat(block.flat_out,sizes)
    assert len(set(block.flat_in))==len(block.flat_in)
    assert len(set(block.flat_out))==len(block.flat_out)
    if not through_B:assert set(block.flat_in)==set(block.flat_out),'pool endpoint permutation'
    headers=[];codes=[];active=[];histories=[]
    for x in xs:
        h=expected(x)[1];u,v,e=q,x>>h,0;packed=0;count=0
        history=[0]*junk_n;used=1;flag_index=None
        for i,p in enumerate(plan):
            count+=int(v+e!=0)
            if p['init']:
                flag_index=used;used+=1;history[flag_index]=int(u>=1<<p['k'])
            cs=carry_oracle(u,v,e,p['carry'])
            history[used:used+len(cs)]=cs;used+=len(cs)
            oldflag=history[flag_index] if p['fmode'] else None
            u,v,e,code=reference_cell(u,v,e);g=(code>>2)&1
            if q==97 and x==65 and i==1 and high_min==1:
                assert oldflag==1 and g==0 and (code>>3)&1==1,('q97 provenance witness',oldflag,code)
            if p['fmode']=='same':
                history[used]=oldflag*g;used+=1
                history[flag_index]=int(u>=1<<p['k'])
            elif p['fmode']=='drop':
                history[used]=oldflag;used+=1
                history[flag_index]=int(u>=1<<(p['k']-1))
            nw=m.widths[i+1] if i+1<T else m.pass_tail_width
            assert max(u,v)<1<<nw,('oracle tail',q,x,i)
            if q==127 and x==21 and i==2:assert code==0
            packed|=code<<(4*i)
        assert (u,v,e)==(1,0,0) and used==junk_n
        history[0]=e
        headers.append(h);codes.append(packed);active.append(count)
        histories.append(sum(b<<i for i,b in enumerate(history)))
    init=sim.pack([q]*len(xs),ins[0])|sim.pack(xs,ins[1][:n])
    want=sim.pack([2*x%q for x in xs] if through_B else [1]*len(xs),outs[0])|sim.pack(codes,outs[1][lent:])|sim.pack(headers,outs[3])|sim.pack(histories,outs[2])
    if through_B:init |= sim.pack([2]*len(xs),ins[5])
    if fault:
        assert fired,('fault did not reach actual gate',fault,q)
        width=max(block.wires|set(block.flat_in)|set(block.flat_out))+1
        forward_bad=bad_wires(sim.run(block.ops,width,init,len(xs)),want)
        rev=array('i',(v for i in range(len(block.ops)-4,-1,-4) for v in block.ops[i:i+4]))
        inverse_bad=bad_wires(sim.run(rev,width,want,len(xs)),init)
        dirty=0
        for state,inv in ((init,False),(want,True)):
            try:release_run(block,state,len(xs),inv)
            except AssertionError:dirty+=1
        assert forward_bad and inverse_bad,('eager fault escaped',fault,q,forward_bad,inverse_bad)
        return dict(fault=fault,q=q,cases=len(xs),forward_bad=forward_bad,inverse_bad=inverse_bad,dirty_release_directions=dirty,fired=len(fired))
    assert bad_wires(release_run(block,init,len(xs)),want)==0,('actual pool',q,cp)
    faults={}
    if inverse:
        assert bad_wires(release_run(block,want,len(xs),True),init)==0,('pool inverse',q)
        D=Circuit();pad=D.alloc(7);spectator=D.alloc(n);regs=[D.alloc(k) for k in sizes];entry=[list(r) for r in regs]
        D.apply(block,regs,inverse=True)
        si=sim.pack([2*x%q for x in xs] if through_B else [1]*len(xs),entry[0])|sim.pack(codes,entry[1][lent:])|sim.pack(headers,entry[3])|sim.pack(histories,entry[2])|sim.pack(xs,spectator)|sim.pack([91]*len(xs),pad)
        sw=sim.pack([q]*len(xs),regs[0])|sim.pack(xs,regs[1][:n])|sim.pack(xs,spectator)|sim.pack([91]*len(xs),pad)
        if through_B:sw |= sim.pack([2]*len(xs),regs[5])
        assert bad_wires(sim.run(D.prims,D.num_wires,si,len(xs)),sw)==0,('fresh mapped pool inverse',q,cp)
        if controls:
            for name,w in [('dirty_H',entry[3][0]),('dirty_G',entry[1][lent]),('dirty_E',entry[2][0])]:
                mutated=dict(si);mutated[w]=mutated.get(w,0)^((1<<len(xs))-1)
                count=bad_wires(sim.run(D.prims,D.num_wires,mutated,len(xs)),sw)
                assert count>0,('pool fault escaped',q,name)
                faults[name]=count
    return dict(q=q,cp=cp,production_cp=0,through_B=through_B,junk=junk_n,carry_limit=carry_limit,high_min=high_min,cases=len(xs),ccx=block.ops[::4].count(CCX),isolated_wires=len(set(block.flat_in)|block.wires),baseline=sum(sizes),tail_boundaries=len(loans),releases=len(block.events),release_scope='composition events plus explicit forward/inverse zero-tail assertions; nested blocks checked separately',forward_bad=0,inverse_bad=0 if inverse else None,controls=faults,max_active=max(active))


class PinSink:
    def __init__(self):
        self.circuit=hashlib.sha256();self.method=hashlib.sha256();self.size=0
        G={'x':hex(SECP256K1_G[0]),'y':hex(SECP256K1_G[1])}
        prefix=json.dumps(dict(method_id='method_shor_secp256k1_canonical',public_key=G,emitted_circuit_digest='0'*64),separators=(',',':'))[:-1]+',"circuit":'
        self.method.update(prefix.encode());self.method_size=len(prefix)
    def write(self,s):
        b=s.encode();self.circuit.update(b);self.method.update(b);self.size+=len(b);return len(s)
    def finish(self):
        rest=dict(universal_certificate=None,wrapper_certificate=None,schedule=None,observations=None,retry_model=None,statistical_binding=None,claims_family=False,claims_secp256k1=True)
        suffix=','+json.dumps(rest,separators=(',',':'))[1:]
        self.method.update(suffix.encode())
        return dict(circuit_sha256=self.circuit.hexdigest(),method_sha256=self.method.hexdigest(),circuit_bytes=self.size,method_bytes=self.method_size+self.size+len(suffix))


def off_pins():
    from qec.emit import write_circuit
    assert not model.JUMP4,'off-pin stage must start in default/off mode'
    from jump_mul_check import flags_and_shapes
    # Startup assertions are fresh processes; full default-off emission is
    # streamed once, without retaining circuit/method duplicates.
    startup={}
    code='from qec import model;print(int(model.JUMP4))'
    for value in (None,'0','1','invalid'):
        env=dict(os.environ)
        if value is None:env.pop('QEC_JUMP4',None)
        else:env['QEC_JUMP4']=value
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
        assert p.returncode!=0 if value=='invalid' else p.returncode==0 and p.stdout.strip()==str(int(value=='1'))
        startup['unset' if value is None else value]=p.returncode
    gen=ControlledPointAdd(SECP256K1,SECP256K1_G);C=gen.build(True);sink=PinSink();emission=write_circuit(sink,C,gen.layout);pins=sink.finish()
    assert pins['circuit_sha256']=='39eafe46cc0efe73a140f7b78b9e6287513d07dabb0af440e7ff330534a242ad',pins
    assert pins['method_sha256']=='6bee6ad72997ee19fbef03b39a481c9b45cfb4d781a64161bb15d9bd27b8e106',pins
    return dict(status='pass',pins=pins,emission=emission,startup=startup,kept_large_outputs=0)


def whole(controls=False):
    import gc
    from check import build_ops,structural,check,special_points
    from jump_mul_check import resources,count_graph
    from unittest.mock import patch
    assert model.JUMP4
    rng=random.Random(8204);rows=[]
    specs=[]
    for bits in (4,6,7):
        curves=toy_secp_curves(bits)[:2]
        if bits==4 and len(curves)==1:
            curves.append(Curve(13,0,2,'second 4-bit prime-order curve'))
        assert len(curves)==2,('missing promised toy curves',bits)
        for curve in curves:
            A=next(P for P in curve.points()[1:] if P[1])
            specs.append((curve,A,[(c,R) for c in (0,1) for R in curve.points()],'all_points'))
    for k in (1,2):
        A=SECP256K1.mul(k,SECP256K1_G)
        pts=special_points(SECP256K1,A)+[SECP256K1.mul(rng.randrange(1,1<<256),A) for _ in range(64)]
        specs.append((SECP256K1,A,[(c,R) for c in (0,1) for R in pts],'special_plus64seed8204'))
    public_calls=[];public=DialogMultiplier.apply
    def traced(self,Xr,Yr,inverse=False,test=None,flag=None):
        assert test is None and flag is None,'Jump kernel used unsupported cut'
        public_calls.append(bool(inverse));return public(self,Xr,Yr,inverse,test,flag)
    def forbidden(*a,**k):raise AssertionError('legacy multiplier body used by Jump kernel')
    for curve,A,cases,label in specs:
        gen=ControlledPointAdd(curve,A)
        with patch.object(DialogMultiplier,'_body',forbidden),patch.object(DialogMultiplier,'apply',traced):
            C,ops,emitted=build_ops(gen,True)
        counts=structural(ops,C.num_wires);assert C.num_wires<=4096
        assert counts['CCX']==count_graph(C.prims,C.calls)
        assert not check(gen,ops,cases),('whole arithmetic',curve.q,A)
        # The independent parser expanded the emitted hierarchy. Compare its
        # complete primitive bytes with a separately built flat circuit.
        h=hashlib.sha256(ops.tobytes()).hexdigest()
        faults={}
        if controls:
            faulty=array('i',ops);x=gen.layout['x'][0];faulty.extend([0,x,x,x])
            fails=check(gen,faulty,cases)
            assert fails,'whole output fault escaped'
            faults['actual_output_X']=len(fails)
            del faulty
        del ops,emitted,C;gc.collect()
        fresh=ControlledPointAdd(curve,A);F,flat,_=build_ops(fresh,False)
        assert h==hashlib.sha256(flat.tobytes()).hexdigest(),('hier/freshflat primitive mismatch',curve.q)
        assert counts==structural(flat,F.num_wires)
        rows.append(dict(q=curve.q,A=A,population=label,lanes=len(cases),counts=counts,wires=F.num_wires,primitive_sha256=h,hier_freshflat_equal=True,controls=faults))
        print(json.dumps(rows[-1]),flush=True)
        del flat,F,fresh,gen;gc.collect()
    assert public_calls and set(public_calls)=={False,True}
    return dict(status='pass',rows=rows,public_calls=len(public_calls),legacy_body_forbidden=True,resources=resources())


def carry_oracle(u,stored,e,count):
    v=stored^e;c=0;out=[]
    for i in range(count+1):
        c=int(((u>>i)&1)+(1-((v>>i)&1))+c>=2)
        if i:out.append(c)
    return out


def high_local(w,k,carry,mode,init):
    C=ReleaseCircuit();m=DialogMultiplier(C,Mod((1<<w)-1))
    block=m._jump4_step(w,carry=carry,kappa=k,high=True,fmode=mode,init=init)
    sizes=[w,w,1,4,0,carry,1,int(mode in ('same','drop')),w-k+1+int(mode!='same')]
    ins=split_flat(block.flat_in,sizes);outs=split_flat(block.flat_out,sizes)
    assert len(block.flat_in)==sum(sizes)
    cases=[(u,v-e,e) for u in range(1,1<<w,2) for v in range(1<<w)
           if not(u>=1<<k and v>=1<<k) for e in ((0,1) if v and v%2==0 else (0,))]
    checked=0
    for at in range(0,len(cases),2048):
        chunk=cases[at:at+2048];nu=[];nv=[];ne=[];code=[];jn=[];flags=[];slots=[]
        for u,v,e in chunk:
            ou,ov,oe,c=reference_cell(u,v,e);g=(c>>2)&1;old=int(u>=1<<k)
            nu.append(ou);nv.append(ov);ne.append(oe);code.append(c)
            jn.append(sum(b<<i for i,b in enumerate(carry_oracle(u,v,e,carry))))
            flags.append(old*(1-g) if mode=='same' else int(ou>=1<<(k-1)) if mode=='drop' else old)
            slots.append(old*g if mode=='same' else old if mode=='drop' else 0)
        u,v,e=zip(*chunk);inputflag=[0 if init else int(x>=1<<k) for x in u]
        si=sim.pack(u,ins[0])|sim.pack(v,ins[1])|sim.pack(e,ins[2])|sim.pack(inputflag,ins[6])
        sw=sim.pack(nu,outs[0])|sim.pack(nv,outs[1])|sim.pack(ne,outs[2])|sim.pack(code,outs[3])|sim.pack(jn,outs[5])|sim.pack(flags,outs[6])|sim.pack(slots,outs[7])
        assert bad_wires(release_run(block,si,len(chunk)),sw)==0,('high local',w,k,carry,mode,init,at)
        assert bad_wires(release_run(block,sw,len(chunk),True),si)==0,('high local inverse',w,k,carry,mode,init,at)
        D=Circuit();spect=D.alloc(3);regs=[D.alloc(x) for x in sizes];end=[list(r) for r in regs];D.apply(block,regs,inverse=True)
        di=sim.pack(nu,end[0])|sim.pack(nv,end[1])|sim.pack(ne,end[2])|sim.pack(code,end[3])|sim.pack(jn,end[5])|sim.pack(flags,end[6])|sim.pack(slots,end[7])|sim.pack([5]*len(chunk),spect)
        dw=sim.pack(u,regs[0])|sim.pack(v,regs[1])|sim.pack(e,regs[2])|sim.pack(inputflag,regs[6])|sim.pack([5]*len(chunk),spect)
        assert bad_wires(sim.run(D.prims,D.num_wires,di,len(chunk)),dw)==0,('high fresh inverse',w,k,carry,mode,init,at)
        checked+=len(chunk)
    return dict(w=w,k=k,carry=carry,mode=mode,init=init,cases=checked,ccx=block.ops[::4].count(CCX),fresh_inverse=True)


def eager_checks(max_q=127,secp=False,controls=False):
    assert callable(getattr(DialogMultiplier,'_jump4_eager_plan',None)),'eager plan absent'
    local=[]
    for w in range(2,9):
        for k in range(1,w):
            for carry in sorted(set([0,min(1,k-1),k-1])):
                for mode in ('same','drop','final'):
                    for init in (False,True):local.append(high_local(w,k,carry,mode,init))
    rows=[];refusals=[]
    # Each population checks exact physical history, through B preservation,
    # observed releases, and a fresh remapped inverse from integer expectations.
    for q in primes(max_q):
        for minimum in (1,DialogMultiplier.HIGH_MIN):
            for budget in (0,1,5,None):
                if q==3 and minimum==1:
                    # Extra QA-only shape beyond the promised 13/31/127
                    # forced corpus has only two future wires for three
                    # clean scratch inputs. Pin its explicit safe refusal.
                    try:check_pool(q,carry_limit=budget,high_min=minimum,through_B=True)
                    except AssertionError as exc:
                        assert exc.args[0][0]=='Jump4 clean scratch shortage',exc
                        refusals.append(dict(q=q,budget=budget,high_min=minimum,reason=str(exc)))
                    else:raise AssertionError('unsafe scratch shortage accepted')
                else:rows.append(check_pool(q,carry_limit=budget,high_min=minimum,through_B=True))
    if secp:
        q=model.SECP256K1_Q;rng=random.Random(8204)
        xs=secp_inputs()+[rng.randrange(1,q) for _ in range(64)]
        for minimum in (1,DialogMultiplier.HIGH_MIN):
            for budget in (0,1,967,None):
                rows.append(check_pool(q,xs=xs,carry_limit=budget,high_min=minimum,through_B=True,controls=controls))
    m=DialogMultiplier(Circuit(True),Mod(model.SECP256K1_Q));plan=m._jump4_eager_plan()
    assert len(plan)==320
    for p in plan:
        assert p['carry']<=max(0,(p['k'] if p['high'] else p['w'])-1)
    flags=sum(int(p['init'])+int(p['fmode'] in ('same','drop')) for p in plan)
    carry=sum(p['carry'] for p in plan)
    assert m._jump4_junk_size()==1+flags+carry<=2236
    # Count actual per-step blocks and whole producers against the identical
    # high/flag schedule with no carries; do not infer the marginal saving.
    from jump_mul_check import count_graph,resources
    costs=[]
    for budget in (0,1,967,None):
        C=Circuit(True);m=DialogMultiplier(C,Mod(model.SECP256K1_Q))
        if budget is not None:m.jump4_carry_limit=budget
        ps=m._jump4_eager_plan();n=m.mod.n
        block=C.record_shape(m._jump4_euclid_pass,[n,n+m._jump4_pool_sizes()[2],m._jump4_junk_size(),3])
        steps=[]
        for p in ps:
            b=m._jump4_step(p['w'],carry=p['carry'],kappa=p['k'] if p['fmode'] else None,high=p['high'],fmode=p['fmode'],init=p['init'])
            steps.append(dict(**p,ccx=count_graph(b.ops,b.calls)))
        costs.append(dict(budget=budget,carry=sum(p['carry'] for p in ps),ccx=count_graph(block.ops,block.calls),steps=steps))
    for c in costs:
        c['saving_vs_same_high_zero_carry']=costs[0]['ccx']-c['ccx']
        assert c['saving_vs_same_high_zero_carry']==c['carry']
    faults=[]
    if controls:
        faults.append(check_pool(97,xs=[65],high_min=1,fault='b_for_g'))
        for f in ('dirty_carry','dirty_flag','skipped_flag_drop','high_bit_omission','wrong_K_minus2','early_future_tape'):
            faults.append(check_pool(127,high_min=1,fault=f))
    whole_resources=resources()
    n=256;T=len(plan)
    # E is INSIDE the envelope. R and S replace Euclid operand space;
    # Coefficient scratch and outside live wires come from actual constructions.
    reader_scratch=whole_resources['phase_shapes']['coefficient_internal']
    inventory=dict(raw_tape=4*T,coefficient_RS=2*n,header=3,E=1,flag_archive=flags,
                   retained_carry=carry,total_junk=1+flags+carry,outer_flags=max(r['outer_live'] for r in whole_resources['multiplier_phase_allocations']),
                   reader_scale_scratch=reader_scratch,actual_whole_peak=whole_resources['wires'],
                   high_scratch_max=max(p['scratch'] for p in plan),
                   high_scratch_source='future clean raw tape then unconsumed junk; never prior history')
    inventory['persistent_reader']=4*T+2*n+3+1+flags+carry+inventory['outer_flags']
    assert inventory['persistent_reader']+inventory['reader_scale_scratch']<=4096
    return dict(status='pass',local=local,rows=rows,safe_refusals=refusals,costs=costs,controls=faults,inventory=inventory,resources=whole_resources)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pool','off-pins','eager','whole'],required=True)
    ap.add_argument('--max-q',type=int,default=127);ap.add_argument('--secp-directed',action='store_true');ap.add_argument('--controls',action='store_true');ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.stage=='off-pins':payload=off_pins()
    elif args.stage=='whole':payload=whole(args.controls)
    elif args.stage=='eager':payload=eager_checks(args.max_q,args.secp_directed,args.controls)
    else:
        assert model.JUMP4
        rows=[check_pool(q,controls=args.controls and q==127) for q in primes(args.max_q)]
        if args.secp_directed:
            q=model.SECP256K1_Q;rng=random.Random(8204)
            rows.append(check_pool(q,xs=secp_inputs()+[rng.randrange(1,q) for _ in range(512)],controls=args.controls))
        payload=dict(rows=rows)
    result=dict(status='pass',stage=args.stage,payload=payload,argv=sys.argv,head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ('qec/model.py','qec/point_add.py','qec/two_bit.py','scripts/jump_pr32_check.py','scripts/jump_step_check.py','scripts/jump_mul_check.py')})
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
