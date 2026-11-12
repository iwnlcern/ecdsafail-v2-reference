"""Independent gate checks of Jump-4 coefficients, scale, reader and rare carries.

Credit ab-godder's arith.add with a one-CCX top bit (PR #9, PR #28).
Expected states use only integer modular arithmetic, never forward output.
"""
from array import array
from unittest.mock import patch
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT),str(Path(__file__).resolve().parent)]
from qec import model,sim
from qec.builder import CCX, Circuit
from qec import two_bit as tb
from qec.two_bit import TwoBitMod
from qec.point_add import DialogMultiplier
from jump_step_check import ReleaseCircuit,release_run,bad_wires,split_flat,primes


def cell_block(q):
    C = ReleaseCircuit()
    mod = TwoBitMod(q)
    n = q.bit_length()
    def body(cell,R,S):
        mod.jump4_cell(C,cell,R,S)
    return C.record_shape(body,[4,n,n])


def scale_block(q):
    C = ReleaseCircuit()
    mod = TwoBitMod(q)
    n = q.bit_length()
    def body(H,S):
        mod.jump4_scale(C,H,S)
    return C.record_shape(body,[3,n])


def run_cases(q,cases,scale=False,block=None):
    block = block or (scale_block(q) if scale else cell_block(q))
    sizes = [3,q.bit_length()] if scale else [4,q.bit_length(),q.bit_length()]
    ins,outs = split_flat(block.flat_in,sizes),split_flat(block.flat_out,sizes)
    for at in range(0,len(cases),4096):
        chunk = cases[at:at+4096]
        if scale:
            ss,hh = zip(*chunk)
            init = sim.pack(hh,ins[0])|sim.pack(ss,ins[1])
            want = sim.pack(hh,outs[0])|sim.pack([(s<<h)%q for s,h in chunk],outs[1])
        else:
            ss,rr,codes = zip(*chunk)
            er,es = [],[]
            for s,r,c in chunk:
                g,b,d = (c>>2)&1,c>>3,1+(c&3)
                z = ((s<<d)+g*r)%q
                a,z = (z,r) if g&b else (r,z)
                er.append(a);es.append(z)
            init = sim.pack(codes,ins[0])|sim.pack(rr,ins[1])|sim.pack(ss,ins[2])
            want = sim.pack(codes,outs[0])|sim.pack(er,outs[1])|sim.pack(es,outs[2])
        actual = release_run(block,init,len(chunk))
        assert bad_wires(actual,want)==0,('coefficient physical mismatch',q,scale,at)
        inverse = release_run(block,want,len(chunk),inverse=True)
        assert bad_wires(inverse,init)==0,('coefficient independent inverse mismatch',q,scale,at)
    wires = len(set(block.flat_in)|block.wires)
    return dict(q=q,scale=scale,cases=len(cases),ccx=block.ops[::4].count(CCX),wires=wires,
                scratch=wires-sum(sizes),releases=len(block.events),forward_bad=0,inverse_bad=0)


def quotient_cases():
    """Directed integer cases for coefficient arithmetic."""
    q = model.SECP256K1_Q
    rng = random.Random(7981)
    cells,scales = [],[]
    for c in range(16):
        g,d = (c>>2)&1,1+(c&3)
        for k in range(1,(1<<d)+g):
            if not g:
                r = rng.randrange(q)
                pairs = [(-(-k*q//(1<<d)),r),((k*q-1)//(1<<d),r)]
            else:
                r = (k*q)%(1<<d)+(1<<d)*rng.randrange(1,(q>>d)-1)
                s = (k*q-r)>>d
                pairs = [(s,r),(s,r+1),(s,r-1)]
            for s,r in pairs:
                assert 0<=s<q and 0<=r<q
                cells.append((s,r,c))
    for h in range(1,5):
        for k in range(1,1<<h):
            scales += [(-(-k*q//(1<<h)),h),((k*q-1)//(1<<h),h)]
    e_cells = [(((s<<(1+(c&3)))+((c>>2)&1)*r)//q
                -(((s<<(1+(c&3)))+((c>>2)&1)*r)>>256)) for s,r,c in cells]
    e_scales = [(s<<h)//q-((s<<h)>>256) for s,h in scales]
    assert (len(cells),sum(e_cells),len(scales),sum(e_scales))==(284,172,52,26)
    return cells,scales,e_cells,e_scales


def check_reader(q,xs=None):
    C = ReleaseCircuit()
    m = DialogMultiplier(C,TwoBitMod(q))
    n,T = q.bit_length(),m.iters
    def body(G,H,R,S):
        m._jump4_bezout_pass(G,H,R,S)
    block = C.record_shape(body,[4*T,3,n,n])
    ins,outs = split_flat(block.flat_in,[4*T,3,n,n]),split_flat(block.flat_out,[4*T,3,n,n])
    cases = [(x,y) for x in (range(1,q) if xs is None else xs) for y in (0,1,2%q,q-1)]
    hs,gs = [],[]
    for x,y in cases:
        h,codes = model.jump4_physical_tape(q,x,T)
        hs.append(h);gs.append(sum(c<<(4*i) for i,c in enumerate(codes)))
        # Terminal/padding identities hold only on the reachable S=0 domain.
        r,s = y,0
        endpoint = True
        for c in reversed(codes):
            if endpoint:
                assert c in (3,7)
                assert s==0, ('reachable padding/terminal domain',q,x,y,c)
                if c==7: endpoint = False
            g,b,d = (c>>2)&1,c>>3,1+(c&3)
            s = ((s<<d)+g*r)%q
            if g&b:
                r,s = s,r
        assert (r,(s<<h)%q)==(0,x*y%q)
    ys = [y for x,y in cases]
    init = sim.pack(gs,ins[0])|sim.pack(hs,ins[1])|sim.pack(ys,ins[2])
    want = sim.pack(gs,outs[0])|sim.pack(hs,outs[1])|sim.pack([x*y%q for x,y in cases],outs[3])
    assert bad_wires(release_run(block,init,len(cases)),want)==0,('reader matrix',q)
    assert bad_wires(release_run(block,want,len(cases),inverse=True),init)==0,('reader inverse',q)
    return dict(q=q,cases=len(cases),forward_bad=0,inverse_bad=0)



def packed_cases(block,q,cases,scale=False):
    sizes = [3,q.bit_length()] if scale else [4,q.bit_length(),q.bit_length()]
    ins,outs = split_flat(block.flat_in,sizes),split_flat(block.flat_out,sizes)
    if scale:
        ss,hh = zip(*cases)
        return (sim.pack(hh,ins[0])|sim.pack(ss,ins[1]),
                sim.pack(hh,outs[0])|sim.pack([(s<<h)%q for s,h in cases],outs[1]))
    ss,rr,cs = zip(*cases)
    pairs = []
    for s,r,c in cases:
        z = ((s<<(1+(c&3)))+((c>>2)&1)*r)%q
        pairs.append((z,r) if c&12==12 else (r,z))
    return (sim.pack(cs,ins[0])|sim.pack(rr,ins[1])|sim.pack(ss,ins[2]),
            sim.pack(cs,outs[0])|sim.pack([r for r,s in pairs],outs[1])
            |sim.pack([s for r,s in pairs],outs[2]))


def mismatch_mask(block,initial,want,lanes,inverse=False):
    ops = block.ops
    if inverse:
        ops = array('i',(v for at in range(len(ops)-4,-1,-4) for v in ops[at:at+4]))
    n = max(block.wires|set(block.flat_in)|set(block.flat_out))+1
    state = sim.run(ops,n,initial,lanes)
    mask = 0
    for w,a in enumerate(state):
        mask |= a^want.get(w,0)
    return mask


def fault_block(name,scale=False):
    """Inject actual gate faults only in checker-owned construction."""
    from contextlib import ExitStack
    R = tb._Jump4Reduction
    with ExitStack() as stack:
        if name=='reversed_rotation':
            original = tb._jump4_rotate
            def rotate(C,*args):
                at = len(C.prims);original(C,*args)
                segment = C.prims[at:]
                C.prims[at:] = array('i',(v for i in range(len(segment)-4,-1,-4) for v in segment[i:i+4]))
            stack.enter_context(patch.object(tb,'_jump4_rotate',rotate))
        elif name=='skipped_K_plus_e':
            original = R.increment
            def increment(self,C,e,K):
                cx = C.cx
                def omit(a,b):
                    if (a,b)!=(e,K[0]): cx(a,b)
                with patch.object(C,'cx',omit): original(self,C,e,K)
            stack.enter_context(patch.object(R,'increment',increment))
        elif name=='skipped_low_clear':
            stack.enter_context(patch.object(R,'clear_low',lambda *args:None))
        elif name=='omitted_top_move':
            stack.enter_context(patch.object(R,'move_top',lambda *args:None))
        elif name=='unconditional_b_swap':
            stack.enter_context(patch.object(tb,'_jump4_swap',lambda C,g,b,R,S:tb.cswap(C,b,R,S)))
        elif name=='dirty_quotient':
            original = R.clear_k
            def dirty(self,C,g,e0,e1,R,S,K):
                original(self,C,g,e0,e1,R,S,K);C.x(K[0])
            stack.enter_context(patch.object(R,'clear_k',dirty))
        elif name in ('dropped_selector_make','dropped_selector_unmake'):
            original = R.and_gate
            calls = 0
            def and_gate(self,C,ops,base,prods,k):
                nonlocal calls
                phase = calls//6%2;calls+=1
                if k==5 and phase==(name=='dropped_selector_unmake'):
                    return
                original(self,C,ops,base,prods,k)
            stack.enter_context(patch.object(R,'and_gate',and_gate))
        else: raise ValueError(name)
        return scale_block(model.SECP256K1_Q) if scale else cell_block(model.SECP256K1_Q)


def check_controls():
    q = model.SECP256K1_Q
    cells,scales,ec,es = quotient_cases()
    # All-codes power/boundary cases also exercise low-clear, top-move and swap.
    population = cells+[(s,r,c) for c in range(16) for s in (0,1,2,q//2,q-2,q-1)
                        for r in (0,1,q//2,q-1)]
    rows = {}
    for name in ('reversed_rotation','skipped_K_plus_e','skipped_low_clear','omitted_top_move',
                 'unconditional_b_swap','dirty_quotient','dropped_selector_make','dropped_selector_unmake'):
        block = fault_block(name)
        cases = cells if name=='skipped_K_plus_e' else population
        initial,want = packed_cases(block,q,cases)
        fm = mismatch_mask(block,initial,want,len(cases))
        im = mismatch_mask(block,want,initial,len(cases),True)
        assert fm and im, ('uncaught actual circuit fault',name,fm,im)
        release_caught = []
        for inv,a,b in ((False,initial,want),(True,want,initial)):
            try: release_run(block,a,len(cases),inverse=inv)
            except AssertionError: release_caught.append(inv)
        if name=='dirty_quotient':
            assert release_caught, 'dirty-only fault must fail real release'
        row = dict(cases=len(cases),forward_bad=fm.bit_count(),inverse_bad=im.bit_count(),
                   forward_mask=hex(fm),inverse_mask=hex(im),dirty_release_directions=release_caught)
        if name=='skipped_K_plus_e':
            expected = sum(e<<i for i,e in enumerate(ec))
            assert fm==im==expected, ('rare middle e1 mask',hex(expected),hex(fm),hex(im))
            sb = fault_block(name,True)
            si,sw = packed_cases(sb,q,scales,True)
            sf,sv = mismatch_mask(sb,si,sw,len(scales)),mismatch_mask(sb,sw,si,len(scales),True)
            expected_s = sum(e<<i for i,e in enumerate(es))
            assert sf==sv==expected_s, ('rare scale e1 mask',hex(expected_s),hex(sf),hex(sv))
            row['scale'] = dict(cases=52,e1=26,forward_mask=hex(sf),inverse_mask=hex(sv))
            row['middle_e1'] = 172
        rows[name] = row
    # Execute the wrong historical order as gates on its documented witness.
    C = ReleaseCircuit();m = TwoBitMod(13)
    def old(cell,R,S):
        e0,e1,g,b = cell
        m.modadd(C,R,S,ctrl=g);tb._jump4_swap(C,g,b,R,S)
        m.moddbl(C,S);m._cdbl(C,e0,S)
        m._cdbl(C,e1,S);m._cdbl(C,e1,S)
    block = C.record_shape(old,[4,4,4])
    h,codes = model.jump4_physical_tape(13,2,DialogMultiplier(C,m).iters)
    r,s = 1,0
    wrong = False
    for c in reversed(codes):
        a,w = packed_cases(block,13,[(s,r,c)])
        wrong |= bool(mismatch_mask(block,a,w,1))
        z = ((s<<(1+(c&3)))+((c>>2)&1)*r)%13
        r,s = (z,r) if c&12==12 else (r,z)
    assert wrong
    rows['old_order_q13_x2_y1'] = dict(caught=True)
    return rows


def check_selectors():
    q = model.SECP256K1_Q
    rows = []
    for scale in (False,True):
      red = tb._Jump4Reduction(q,scale)
      for chain in range(2):
        C = ReleaseCircuit()
        def body(K,A):
            (one,) = C.alloc(1);C.x(one)
            sels,masks,made = red.make_selectors(C,K+[None] if scale else K,chain,one)
            for t,mask in zip(sels,masks):
                for j in range(red.w):
                    if mask>>j&1: C.cx(t,A[j])
            red.unmake(C,made);C.x(one);C.free([one])
        block = C.record_shape(body,[4 if scale else 5,red.w])
        ki,ai = split_flat(block.flat_in,[4 if scale else 5,red.w]);ko,ao = split_flat(block.flat_out,[4 if scale else 5,red.w])
        ks = list(range(16 if scale else 17));lanes=len(ks)
        initial = sim.pack(ks,ki)
        want = sim.pack(ks,ko)|sim.pack([(k+(chain==0))*red.f for k in ks],ao)
        assert bad_wires(release_run(block,initial,lanes),want)==0
        assert bad_wires(release_run(block,want,lanes,True),initial)==0
        rows.append(dict(chain=chain,scale=scale,cases=lanes,ccx=block.ops[::4].count(CCX),clean=True))
    rejected = 0
    bad_tables = [tuple(tuple(((0,None),(0,None)) for _ in range(6)) for _ in range(2)),
                  (tuple([((1,0),(1,None))]+list(tb._JUMP4_SELECTOR_TABLES[0][1:])),tb._JUMP4_SELECTOR_TABLES[1])]
    for tables in bad_tables:
        try: tb._Jump4Reduction(q,tables=tables)
        except ValueError: rejected+=1
    assert rejected==2
    return dict(rows=rows,rejected_nonspanning_or_malformed=rejected)


def reduction_block(q,fault=None,scale=False):
    C=ReleaseCircuit();red=tb._Jump4Reduction(q,scale)
    assert hasattr(red,'quiet_runs'),'quiet-run chains absent'
    if fault=='span':red.quiet_runs=[(1,red.w)]
    elif fault=='mask':
        masks=list(red.masks[0]);masks[0]^=1;red.masks[0]=tuple(masks)
    def body(word):
        S,K=red.reduce(C,word);word[:]=S+K
    return C.record_shape(body,[q.bit_length()+(4 if scale else 5)]),red


def check_quiet_chains(controls=False):
    rows=[];faults={}
    for q in (31,127,991,4079,model.SECP256K1_Q):
      for scale in (False,True):
        block,red=reduction_block(q,scale=scale);n=q.bit_length();N=1<<n
        eligible={i for i in range(1,red.w) if all((((k+(chain==0))*red.f)>>i)&1==0 for chain in (0,1) for k in range(17))}
        expected=[];i=1
        while i<red.w:
            j=i
            while j<red.w and j in eligible:j+=1
            if j-i>=2:expected.append((i,j))
            i=max(i+1,j)
        assert red.quiet_runs==expected,('quiet spans incomplete or unsound',q,red.quiet_runs,expected)
        if q==model.SECP256K1_Q:
            assert expected,'secp quiet optimization missing'
            rng=random.Random(8303);ls={0,1,(1<<red.w)-1}
            for k in range(18):
                for d in (-1,0,1):
                    ls.add((k*red.f+d)%(1<<red.w))
                    ls.add((-k*red.f+d)%(1<<red.w))
            for a,b in expected:
                for t in (a,b-1):
                    for d in (-1,0,1):ls.add(((1<<t)+d)%(1<<red.w))
            hs={0,1,(1<<(n-red.w))-2,(1<<(n-red.w))-1}
            vals=sorted({k*N+(h<<red.w)+l for k in range(17) for h in hs for l in ls if k*N+(h<<red.w)+l<(16 if scale else 17)*q})
            vals+= [rng.randrange((16 if scale else 17)*q) for _ in range(512)]
        else:vals=list(range((16 if scale else 17)*q))
        for offset in range(0,len(vals),2048):
            vs=vals[offset:offset+2048]
            init=sim.pack(vs,block.flat_in);want=sim.pack([v%q+((v//q)<<n) for v in vs],block.flat_out)
            assert bad_wires(release_run(block,init,len(vs)),want)==0,('quiet reduction',q,offset)
            assert bad_wires(release_run(block,want,len(vs),True),init)==0,('quiet reduction inverse',q,offset)
            D=Circuit();spect=D.alloc(7);r=[D.alloc(n+(4 if scale else 5))];entry=list(r[0]);D.apply(block,r,inverse=True)
            si=sim.pack([v%q+((v//q)<<n) for v in vs],entry)|sim.pack([83]*len(vs),spect)
            sw=sim.pack(vs,r[0])|sim.pack([83]*len(vs),spect)
            assert bad_wires(sim.run(D.prims,D.num_wires,si,len(vs)),sw)==0,('fresh quiet inverse',q,offset)
        rows.append(dict(q=q,scale=scale,cases=len(vals),spans=expected,ccx=block.ops[::4].count(CCX),releases=len(block.events),fresh_inverse=True))
        if controls and q==model.SECP256K1_Q:
            for fault in ('span','mask'):
                broken,_=reduction_block(q,fault,scale)
                initial=sim.pack(vals,broken.flat_in)
                want=sim.pack([v%q+((v//q)<<n) for v in vals],broken.flat_out)
                fm=mismatch_mask(broken,initial,want,len(vals));im=mismatch_mask(broken,want,initial,len(vals),True)
                assert fm and im,('quiet fault escaped',fault)
                faults[('scale_' if scale else 'cell_')+fault]=dict(forward_bad=fm.bit_count(),inverse_bad=im.bit_count())
    return dict(rows=rows,controls=faults)


def check_first_cell(controls=False):
    rows=[]
    for q in (3,13,31,127,model.SECP256K1_Q):
        C=ReleaseCircuit();m=DialogMultiplier(C,TwoBitMod(q));n=q.bit_length();calls=[];apply=C.apply
        def observe(block,regs,inverse=False):
            if len(regs)==3 and [len(r) for r in regs]==[4,n,n]:calls.append(block)
            apply(block,regs,inverse)
        C.apply=observe
        C.record_shape(m._jump4_bezout_pass,[4*m.iters,3,n,n])
        assert len(calls)==m.iters
        first=calls[0]
        assert first.ops[::4].count(CCX)==n,'first reverse S0 specialization absent'
        assert all(b.ops[::4].count(CCX)>n for b in calls[1:]),'specialization escaped first reverse cell'
        rs=list(range(q)) if q<=127 else [0,1,2,q//2,q-2,q-1]+[1<<k for k in range(256)]
        cases=[(0,r,c) for c in (3,7) for r in rs]
        initial,want=packed_cases(first,q,cases)
        assert bad_wires(release_run(first,initial,len(cases)),want)==0
        assert bad_wires(release_run(first,want,len(cases),True),initial)==0
        # Independently remap inverse inputs with spectators from the canonical map.
        D=Circuit();spect=D.alloc(7);regs=[D.alloc(4),D.alloc(n),D.alloc(n)];entry=[list(r) for r in regs]
        D.apply(first,regs,inverse=True)
        cs=[c for _,r,c in cases];rr=[r for _,r,c in cases];ss=[r if c==7 else 0 for _,r,c in cases]
        si=sim.pack(cs,entry[0])|sim.pack(rr,entry[1])|sim.pack(ss,entry[2])|sim.pack([83]*len(cases),spect)
        sw=sim.pack(cs,regs[0])|sim.pack(rr,regs[1])|sim.pack([83]*len(cases),spect)
        assert bad_wires(sim.run(D.prims,D.num_wires,si,len(cases)),sw)==0
        rows.append(dict(q=q,cases=len(cases),ccx=n,first_only=True,fresh_inverse=True))
    caught=False
    if controls:
        def wrong(self,C,cell,R,S):
            for r,s in zip(R,S):C.ccx(cell[2],r,s)
        with patch.object(TwoBitMod,'jump4_cell',wrong):
            try:check_reader(13,[2])
            except AssertionError:caught=True
        assert caught,'ordinary-cell specialization escaped reachable reader check'
    return dict(rows=rows,ordinary_cell_control=caught if controls else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-q',type=int,default=127)
    ap.add_argument('--optimized-q',type=int,nargs='+',default=[31,127,991,4079])
    ap.add_argument('--r-samples',type=int,default=4)
    ap.add_argument('--secp-directed',action='store_true')
    ap.add_argument('--secp-samples',type=int,default=2048)
    ap.add_argument('--seed',type=int,default=7979)
    ap.add_argument('--controls',action='store_true')
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    assert os.environ.get('QEC_JUMP4')=='1', 'explicit QEC_JUMP4=1 required'
    model.JUMP4 = True
    rows,scale_rows,reader_rows = [],[],[]
    for q in primes(args.max_q):
        cases = [(s,r,c) for s in range(q) for r in range(q) for c in range(16)]
        rows.append(run_cases(q,cases))
        scale_rows.append(run_cases(q,[(s,h) for s in range(q) for h in range(5)],scale=True))
        reader_rows.append(check_reader(q))
        print(json.dumps(dict(small_q=q,cell_cases=len(cases),status='pass')),flush=True)
    rng = random.Random(args.seed)
    for q in args.optimized_q:
        if q<=args.max_q:
            continue
        cases = [(s,r,c) for s in range(q) for r in [0,1,q-1]+[rng.randrange(q) for _ in range(args.r_samples)]
                 for c in range(16)]
        rows.append(run_cases(q,cases))
        scale_rows.append(run_cases(q,[(s,h) for s in range(q) for h in range(5)],scale=True))
        print(json.dumps(dict(optimized_q=q,cell_cases=len(cases),status='pass')),flush=True)
    if args.secp_directed:
        q = model.SECP256K1_Q
        boundaries = sorted({z for k in range(256) for base in (1<<k,q-(1<<k))
                             for z in (base-1,base,base+1) if 0<=z<q}|{0,1,q-1})
        cases = [(s,r,c) for c in range(16) for s,r in
                 [(x,0) for x in boundaries]+[(x,q-1) for x in boundaries]
                  ]
        cases += [(rng.randrange(q),rng.randrange(q),i%16) for i in range(args.secp_samples)]
        rows.append(run_cases(q,cases))
        scale_rows.append(run_cases(q,[(s,h) for h in range(5) for s in boundaries]
                                    +[(rng.randrange(q),i%5) for i in range(505)],scale=True))
        reader_rows.append(check_reader(q,[x for x in boundaries if x]+[rng.randrange(1,q) for _ in range(128)]))
        cells,scales,ec,es = quotient_cases()
        rows.append(run_cases(q,cells))
        scale_rows.append(run_cases(q,scales,scale=True))
    result = dict(status='pass',rows=rows,scale_rows=scale_rows,reader_rows=reader_rows,
                  controls=check_controls() if args.controls else None,selectors=check_selectors(),
                  quiet=check_quiet_chains(args.controls),first=check_first_cell(args.controls),
                  flag=1,rare_quotient=dict(middle=284,middle_e1=172,scale=52,scale_e1=26),
                  head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                  source={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                          for f in ('qec/model.py','qec/two_bit.py','qec/point_add.py','scripts/jump_cell_check.py','scripts/jump_step_check.py')})
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'cell.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
