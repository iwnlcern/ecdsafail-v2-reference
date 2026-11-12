"""Independent Jump-4 codec, offset producer, release and recorded replay checks."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import copy
import random

sys.path[:0] = [str(Path(__file__).resolve().parent.parent),str(Path(__file__).resolve().parent)]
from qec import model, sim
from qec.arith import Mod
from qec.builder import Circuit, CCX
from qec.point_add import DialogMultiplier


def primes(limit):
    return [q for q in range(3, limit + 1, 2)
            if all(q % d for d in range(2, int(q**0.5) + 1))]


def expected(x):
    d = min(4, (x & -x).bit_length() - 1)
    return x >> d, d


def bad_wires(actual, want):
    mask = 0
    for w, value in enumerate(actual):
        mask |= value ^ want.get(w, 0)
    return mask.bit_count()


def check_initial(q, cp, control=None, xs=None):
    C = ReleaseCircuit()
    m = DialogMultiplier(C, Mod(q))
    n = q.bit_length()
    U, V, H, copies = C.alloc(n), C.alloc(n), C.alloc(3), C.alloc(cp)
    spectator = C.alloc(n)
    regs = [V, H, copies]
    entry = list(V), list(H), list(copies)
    block = m._jump4_initial_strip(cp)
    C.apply(block, regs)
    xs = list(range(1, q)) if xs is None else xs
    vals, headers = zip(*(expected(x) for x in xs))
    bi = split_flat(block.flat_in,[n,3,cp])
    bo = split_flat(block.flat_out,[n,3,cp])
    local_in = sim.pack(xs,bi[0])
    local_out = sim.pack(vals,bo[0])|sim.pack(headers,bo[1])
    assert bad_wires(release_run(block,local_in,len(xs)),local_out)==0
    assert bad_wires(release_run(block,local_out,len(xs),inverse=True),local_in)==0
    initial = sim.pack(xs, entry[0]) | sim.pack([q] * len(xs), U) | sim.pack(xs, spectator)
    want = (sim.pack(vals, V) | sim.pack(headers, H) | sim.pack([q] * len(xs), U)
            | sim.pack(xs, spectator))
    ops = array('i', C.prims)
    if control == 'wrong_header':
        ops.extend((0, H[0], H[0], H[0]))
    elif control == 'dropped_rotation':
        # A Fredkin's CX/CCX/CX triple; omit one actual controlled reflection swap.
        pos = next(i for i in range(4, len(ops) - 4, 4)
                   if ops[i] == CCX and ops[i-4] == 1 and ops[i+4] == 1
                   and ops[i-4:i] == ops[i+4:i+8]
                   and ops[i-3] == ops[i+3] and ops[i-2] == ops[i+2])
        del ops[pos-4:pos+8]
    result = sim.run(ops, C.num_wires, initial, len(xs))
    bad = bad_wires(result, want)
    if control:
        return bad
    assert bad == 0, ('initial physical mismatch/dirty scratch', q, cp, bad)
    # Inverse starts from independently packed output on a fresh mapping,
    # not from the result of the forward simulator.
    D = Circuit()
    DV, DH, DC, DU, spectator2 = D.alloc(n), D.alloc(3), D.alloc(cp), D.alloc(n), D.alloc(n)
    old_out = list(DV), list(DH), list(DC)
    inv_regs = [DV, DH, DC]
    D.apply(block, inv_regs, inverse=True)
    inv_in = (sim.pack(vals, old_out[0]) | sim.pack(headers, old_out[1])
              | sim.pack([q] * len(xs), DU) | sim.pack(xs, spectator2))
    inv_want = (sim.pack(xs, DV) | sim.pack([q] * len(xs), DU) | sim.pack(xs, spectator2))
    inv = sim.run(D.prims, D.num_wires, inv_in, len(xs))
    assert bad_wires(inv, inv_want) == 0, ('inverse physical mismatch/dirty scratch', q, cp)
    assert set(headers) <= set(range(5))
    return dict(q=q, cp=cp, cases=len(xs), header_values=sorted(set(headers)),
                ccx=C.prims[::4].count(CCX), wires=C.num_wires, isolated_wires=C.num_wires-2*n,
                layout=dict(U=U,V=V,H=H,copies=copies,spectator=spectator),
                forward_bad=0,inverse_bad=0,
                digest=hashlib.sha256(str((vals,headers)).encode()).hexdigest())


def check_guard(integrated=False):
    C = Circuit()
    m = DialogMultiplier(C, Mod(7))
    if integrated:
        m.apply(C.alloc(3),C.alloc(3))
        return
    try:
        m.apply(C.alloc(3), C.alloc(3))
    except NotImplementedError as exc:
        assert 'coefficient' in str(exc).lower(), ('guard message', str(exc))
    else:
        raise AssertionError('missing coefficient integration must refuse full apply')


def reference_cell(u, stored, e):
    """Independent exact true-value oracle; no builder/model calls."""
    v = stored + e
    g = v % 2
    swap = int(bool(g) and u > v)
    b = swap if g else e
    if swap:
        u, v = v, u
    v -= g * u
    d = min(4, (v & -v).bit_length()-1) if v else 4
    v //= 2**d
    out_e = int(bool((swap or e) and d == 4 and v and v % 2 == 0))
    return u, v-out_e, out_e, (d-1)+4*g+8*b


def check_codec_reference(max_q):
    for code in range(16):
        g, b, d = model.jump4_decode_cell(code)
        assert (g,b,d) == ((code>>2)&1, code>>3, 1+(code&3))
        assert model.jump4_encode_cell(g,b,d) == code
    for code in (-1,16,100):
        try:
            model.jump4_decode_cell(code)
        except ValueError:
            pass
        else:
            raise AssertionError(('invalid code accepted',code))
    for args in ((2,0,1),(0,-1,1),(0,0,0),(0,0,5)):
        try:
            model.jump4_encode_cell(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(('invalid cell accepted',args))
    codes_seen, cases, transitions = set(), 0, set()
    populations = [(q,range(1,q)) for q in primes(max_q)]
    sq = model.SECP256K1_Q
    populations.append((sq,[sq-512,1,sq-1]+[1<<k for k in range(256)]))
    for q, xs in populations:
        ticks = model.tick_bound(q)
        for x in xs:
            h, codes = model.jump4_physical_tape(q,x,ticks)
            assert (x>>h,h) == expected(x)
            assert len(codes) == ticks
            u,v,e = q,x>>h,0
            semantic_u, semantic_v = q,x
            previous_d = h
            for code in codes:
                iu,iv,ie = u,v,e
                u,v,e,want = reference_cell(u,v,e)
                assert code == want, ('physical codec',q,x,code,want)
                transitions.add((ie,e))
                codes_seen.add(code)
                g,b,d = model.jump4_decode_cell(code)
                # Undo exact relation without using the model reference.
                a,z = u, ((v+e)<<d)+g*u
                if g and b:
                    a,z = z,a
                ei = (1-g)*b
                assert (a,z-ei,ei) == (iu,iv,ie)
                if semantic_v:
                    sd = min(4,(semantic_v & -semantic_v).bit_length()-1)
                    assert sd == previous_d
                    semantic_v >>= sd
                    sg = semantic_v&1
                    ss = int(bool(sg) and semantic_u>semantic_v)
                    assert (sg,ss) == (g,g&b)
                    if ss:
                        semantic_u,semantic_v = semantic_v,semantic_u
                    semantic_v -= sg*semantic_u
                else:
                    assert (g,b,d,e) == (0,0,4,0)
                assert (u,(v+e)<<d) == (semantic_u,semantic_v)
                previous_d = d
            assert (u,v,e) == (1,0,0)
            cases += 1
    assert transitions >= {(0,0),(0,1),(1,0),(1,1)}, transitions
    return dict(cases=cases,codes_seen=sorted(codes_seen),offset_transitions=sorted(transitions))


class ReleaseCircuit(Circuit):
    """Observe real builder releases/allocations without changing production IR."""
    def __init__(self):
        super().__init__()
        self.events = []
        self.tails = []
        self.full_operands = None

    def alloc(self,k=1):
        wires = super().alloc(k)
        self.events.append((len(self.prims), 'alloc', list(wires)))
        return wires

    def free(self,wires):
        self.events.append((len(self.prims), 'free', list(wires)))
        super().free(wires)

    def record(self,fn,regs):
        start, mark = len(self.prims), len(self.events)
        block = super().record(fn,regs)
        block.events = [(p-start,k,w) for p,k,w in self.events[mark:]]
        del self.events[mark:]
        return block

    def record_shape(self,fn,sizes):
        mark = len(self.events)
        block = super().record_shape(fn,sizes)
        # Placeholder allocation is speculative: record_shape restores the
        # allocator. Only its returned block's internal events are physical.
        del self.events[mark:]
        return block

    def apply(self,block,regs,inverse=False):
        if self.full_operands is not None and len(regs)==5:
            w = len(regs[0])
            U,V = self.full_operands
            self.tails.append((len(self.prims),list(U[w:])+list(V[w:])))
        super().apply(block,regs,inverse)


def release_run(block, initial, lanes, inverse=False):
    """Check every actual free forward, every allocation backward, even reused IDs."""
    ops = block.ops
    n = max(block.wires | set(block.flat_in) | set(block.flat_out)) + 1
    if inverse:
        ops = array('i', (v for i in range(len(ops)-4,-1,-4) for v in ops[i:i+4]))
    points = {}
    for pos,kind,wires in block.events:
        if kind == ('alloc' if inverse else 'free'):
            at = len(ops)-pos if inverse else pos
            points.setdefault(at,[]).extend(wires)
    state = [initial.get(w,0) for w in range(n)]
    last = 0
    for pos in sorted(points):
        state = sim.run(ops[last:pos],n,dict(enumerate(state)),lanes)
        assert all(state[w] == 0 for w in points[pos]), ('dirty release',inverse,pos,points[pos])
        last = pos
    return sim.run(ops[last:],n,dict(enumerate(state)),lanes)


def split_flat(flat,sizes):
    out,offset = [],0
    for size in sizes:
        out.append(flat[offset:offset+size]); offset += size
    return out


def step_cases(w):
    return [(u,v-e,e) for u in range(1,2**w,2) for v in range(2**w)
            for e in ((0,1) if v>0 and v%2==0 else (0,))]


def secp_inputs():
    q = model.SECP256K1_Q
    rng = random.Random(23)
    return ([1,q-1,q-512]+[1<<k for k in range(256)]+[q-(1<<k) for k in range(256)]
            +[rng.randrange(1,q) for _ in range(128)])


def check_step(w,cp,cases=None,first=False,controls=False):
    cases = step_cases(w) if cases is None else cases
    C = ReleaseCircuit()
    m = DialogMultiplier(C,Mod((1<<w)-1))
    chain_points = []
    shared = m._shared_difference
    def observed_chain(U,V,g,b,*rest,**kw):
        shared(U,V,g,b,*rest,**kw)
        chain_points.append((len(C.prims),list(U),list(V),g,b))
    m._shared_difference = observed_chain
    block = m._jump4_step(w,first=first,cp=cp)
    sizes = [w,w,1,4,cp]
    U,V,E,G,copies = split_flat(block.flat_in,sizes)
    OU,OV,OE,OG,OC = split_flat(block.flat_out,sizes)
    us,vs,es = zip(*cases)
    targets = [reference_cell(*case) for case in cases]
    tus,tvs,tes,tgs = zip(*targets)
    init = sim.pack(us,U)|sim.pack(vs,V)|sim.pack(es,E)
    want = sim.pack(tus,OU)|sim.pack(tvs,OV)|sim.pack(tes,OE)|sim.pack(tgs,OG)
    point,cu,cv,cg,cb = chain_points[0]
    chain_u,chain_v,chain_g,chain_b = [],[],[],[]
    for iu,iv,ie in cases:
        true_v = iv+ie
        g = true_v&1
        swap = int(bool(g) and iu>true_v)
        a,z = (true_v,iu) if swap else (iu,true_v)
        z -= g*a
        z -= 2*(swap if g else ie)
        assert 0<=z<2**w and z%2==0
        chain_u.append(a);chain_v.append(z);chain_g.append(g);chain_b.append(swap)
    chain_want = (sim.pack(chain_u,cu)|sim.pack(chain_v,cv)|sim.pack(chain_g,[cg])
                  |sim.pack(chain_b,[cb])|sim.pack(es,E))
    assert bad_wires(sim.run(block.ops[:point],max(block.wires)+1,init,len(cases)),chain_want)==0
    actual = release_run(block,init,len(cases))
    bad = bad_wires(actual,want)
    assert bad == 0, ('step mismatch',w,cp,first,bad,
                      next((cases[i],targets[i]) for i in range(len(cases))
                           if any((actual[j]^want.get(j,0))>>i&1 for j in range(len(actual)))))
    inverse = release_run(block,want,len(cases),inverse=True)
    assert bad_wires(inverse,init) == 0, ('step inverse mismatch',w,cp)
    # Fresh physical IDs, arbitrary spectator, independent output packing.
    D = Circuit()
    spectator = D.alloc(w+3)
    regs = [D.alloc(k) for k in sizes]
    entry = [list(r) for r in regs]
    D.apply(block,regs,inverse=True)
    inv_init = (sim.pack(tus,entry[0])|sim.pack(tvs,entry[1])|sim.pack(tes,entry[2])
                |sim.pack(tgs,entry[3])|sim.pack(us,spectator))
    inv_want = sim.pack(us,regs[0])|sim.pack(vs,regs[1])|sim.pack(es,regs[2])|sim.pack(us,spectator)
    assert bad_wires(sim.run(D.prims,D.num_wires,inv_init,len(cases)),inv_want)==0
    faults = {}
    control_evidence = {}
    if controls:
        ops = block.ops
        mutants = {}
        for name,gate in [('ignore_E_parity',(1,E[0],V[0],V[0])),
                          ('archive_E_out',(1,E[0],G[3],G[3]))]:
            pos = next(i for i in range(0,len(ops),4) if tuple(ops[i:i+4])==gate)
            mutant = array('i',ops); del mutant[pos:pos+4]
            if name == 'archive_E_out':
                mutant.extend((0,OG[2],OG[2],OG[2],2,OG[2],OE[0],OG[3],
                               0,OG[2],OG[2],OG[2]))
            mutants[name] = mutant
        mutants['wrong_D'] = ops + array('i',(0,OG[0],OG[0],OG[0]))
        # Omit the real final high-bit write in the shared difference.
        # Require lanes where this actually loses a one, then fails downstream.
        chain_end = chain_points[0][0]
        pos = max(i for i in range(0,chain_end,4)
                  if ops[i]==1 and ops[i+2]==V[-1])
        mutant = array('i',ops); del mutant[pos:pos+4]
        mutants['lost_top'] = mutant
        correct_chain = sim.run(ops[:chain_end],len(actual),init,len(cases))
        wrong_chain = sim.run(mutant[:chain_end-4],len(actual),init,len(cases))
        wrong_final = sim.run(mutant,len(actual),init,len(cases))
        wrong_mask = 0
        for wire,value in enumerate(wrong_final):
            wrong_mask |= value^want.get(wire,0)
        lost = (correct_chain[V[-1]] & ~wrong_chain[V[-1]] & wrong_mask
                & (init.get(U[-1],0)|init.get(V[-1],0)))
        assert lost, ('no active top-bit-loss witness',w,cp,pos)
        lane = (lost & -lost).bit_length()-1
        control_evidence['lost_top'] = dict(
            lost_high_bit_lanes=lost.bit_count(),omitted_primitive_index=pos//4,
            omitted_gate=list(ops[pos:pos+4]),component='shared difference final top-bit write',
            target=V[-1],witness_lane=lane,input_state=cases[lane],expected_output=targets[lane],
            correct_chain_top=(correct_chain[V[-1]]>>lane)&1,
            mutated_chain_top=(wrong_chain[V[-1]]>>lane)&1,
            downstream_mismatch_lanes=wrong_mask.bit_count())
        # Wrong extra half, and early offset clear before the real normalization.
        mutant = array('i',ops)
        for a,b0 in zip(OV,OV[1:]):
            mutant.extend((3,a,b0,b0))
        mutants['mandatory_extra_half'] = mutant
        norm = len(ops)-16  # final Fredkin (3 gates), then g&b low toggle
        mutants['early_E_clear'] = ops[:norm]+array('i',(1,G[3],E[0],E[0]))+ops[norm:]
        # Remove a real known-one low-bit clear (CCX into V after the chain).
        parity_pos = next(i for i in range(0,len(ops),4)
                          if tuple(ops[i:i+4])==(1,E[0],G[3],G[3]))
        pos = next(i for i in range(parity_pos+4,len(ops),4)
                   if ops[i]==CCX and ops[i+3] in V[:min(w,3)])
        mutant = array('i',ops); del mutant[pos:pos+4]
        mutants['known_one_clear'] = mutant
        tmp = next(wires[0] for _,kind,wires in block.events if kind=='free')
        mutants['dirty_scratch'] = ops+array('i',(0,tmp,tmp,tmp))
        if cp:
            mutants['dirty_copy'] = ops+array('i',(0,OC[0],OC[0],OC[0]))
        for name,mutant in mutants.items():
            count = bad_wires(sim.run(mutant,len(actual),init,len(cases)),want)
            assert count>0, ('control escaped',name)
            faults[name] = count
        # Execute a deliberately wrong inverse decoder over real g0/b1
        # outputs. Compare its reconstructed input to independently packed
        # original states, rather than merely counting available code lanes.
        wrong_inputs = []
        for lane,((a,z,eo,code),source) in enumerate(zip(targets,cases)):
            g,b,d = (code>>2)&1,code>>3,1+(code&3)
            if g or not b:
                continue
            a,z = a,((z+eo)<<d)+g*a
            a,z = z,a  # WRONG: unconditional b, instead of g AND b
            reconstructed = a,z-(1-g)*b,(1-g)*b
            if reconstructed != source:
                wrong_inputs.append((lane,reconstructed))
        assert wrong_inputs, 'wrong unconditional-b decoder escaped'
        lane,reconstructed = wrong_inputs[0]
        faults['unconditional_b_swap'] = len(wrong_inputs)
        control_evidence['unconditional_b_swap'] = dict(
            wrong_reconstructions=len(wrong_inputs),witness_lane=lane,
            original_input=cases[lane],physical_output=targets[lane],
            wrong_decoded_input=reconstructed,mutated_rule='swap=b rather than g AND b')
        # A final-state-only check would miss a temporary dirtied exactly
        # across a free boundary and repaired immediately afterwards.
        point,_,wires = next(event for event in block.events if event[1]=='free')
        tmp = wires[0]
        mutant = copy.copy(block)
        flip = array('i',(0,tmp,tmp,tmp))
        mutant.ops = ops[:point]+flip+flip+ops[point:]
        mutant.events = [(p+4 if p==point and kind=='free' else p+8 if p>=point else p,kind,ws)
                         for p,kind,ws in block.events]
        assert bad_wires(sim.run(mutant.ops,len(actual),init,len(cases)),want)==0
        try:
            release_run(mutant,init,len(cases))
        except AssertionError as exc:
            assert exc.args[0][0]=='dirty release'
            faults['dirty_release_repaired_later'] = len(cases)
        else:
            raise AssertionError('release-only fault escaped')
        assert control_evidence.get('lost_top',{}).get('lost_high_bit_lanes',0)>0, 'missing real top-loss evidence'
        assert control_evidence.get('unconditional_b_swap',{}).get('wrong_reconstructions',0)>0, 'missing executed decoder evidence'
    return dict(w=w,cp=cp,first=first,cases=len(cases),ccx=block.ops[::4].count(CCX),
                isolated_wires=len(set(block.flat_in)|block.wires),
                codes_seen=sorted(set(tgs)),
                offset_transitions=sorted(set(zip(es,tes))),
                events=len(block.events),forward_bad=0,inverse_bad=0,controls=faults,
                control_evidence=control_evidence)


def check_step_pass(args):
    rows = [check_step(w,cp,controls=args.controls and w==args.max_width)
            for w in range(1,args.max_width+1) for cp in (0,1,4)]
    pass_rows = [check_pass(q,cp) for q in primes(args.max_q) for cp in (0,1,4)]
    secp_rows = []
    if args.secp_directed:
        q = model.SECP256K1_Q
        xs = secp_inputs()
        secp_rows = [check_pass(q,cp,xs=xs) for cp in (0,1,4)]
        for w in (100,200,256):
            limit = 1<<w
            cases = [(u,v-e,e) for u in (1,3,limit-1,limit//2+1)
                     for v in (0,1,2,16,32,limit-1,limit-2,limit//2)
                     for e in ((0,1) if v and v%2==0 else (0,))]
            rows.extend(check_step(w,cp,cases=cases) for cp in (0,1,4))
    first_rows = [check_step(q.bit_length(),cp,cases=[(q,x>>expected(x)[1],0)
                   for x in range(1,q)],first=True)
                  for q in primes(args.max_q) for cp in (0,1,4)]
    return dict(rows=rows,first_rows=first_rows,pass_rows=pass_rows,secp_rows=secp_rows,status='pass')


def check_pass(q,cp,xs=None,inverse=False,controls=False):
    from jump_pr32_check import check_pool
    return check_pool(q,cp,xs,inverse,controls)


def check_replay(args):
    rows = [check_pass(q,cp,inverse=True,controls=args.controls and q==127 and cp==4)
            for q in primes(args.max_q) for cp in (0,1,4)]
    q = model.SECP256K1_Q
    xs = secp_inputs()
    secp = [check_pass(q,cp,xs=xs,inverse=True,controls=args.controls and cp==4)
            for cp in (0,1,4)] if args.secp_directed else []
    check_guard(args.integrated)
    return dict(stage='replay-budget',rows=rows,secp_rows=secp,status='pass')


def source_identity():
    root = Path(__file__).resolve().parent.parent
    return dict(head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                source={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                        for name in ('qec/model.py','qec/point_add.py','qec/two_bit.py','scripts/jump_step_check.py')},
                command=sys.argv,python=sys.version,flag=int(model.JUMP4))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['initial','step-pass','replay-budget','all'], required=True)
    ap.add_argument('--integrated',action='store_true')
    ap.add_argument('--max-width', type=int, default=8)
    ap.add_argument('--secp-directed', action='store_true')
    ap.add_argument('--max-q', type=int, default=127)
    ap.add_argument('--controls', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    model.JUMP4 = True
    if args.stage=='all':
        summaries=[]
        for stage in ('initial','step-pass','replay-budget'):
            command=['--stage',stage,
                     '--max-width',str(args.max_width),'--max-q',str(args.max_q),'--out',str(args.out/stage)]
            if args.controls:command.append('--controls')
            if args.secp_directed:command.append('--secp-directed')
            if args.integrated:command.append('--integrated')
            main(command)
            receipt=json.loads((args.out/stage/(stage+'.json')).read_text())
            assert receipt['status']=='pass'
            summaries.append(dict(stage=stage,status=receipt['status'],flag=receipt['flag']))
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'all.json').write_text(json.dumps(dict(status='pass',stages=summaries,**source_identity()),indent=2)+'\n')
        return
    if args.stage != 'initial':
        result = check_step_pass(args) if args.stage=='step-pass' else check_replay(args)
        result.update(source_identity())
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/(args.stage+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    codec = check_codec_reference(args.max_q)
    assert callable(getattr(DialogMultiplier, '_jump4_initial_strip', None)), 'initial producer is absent'
    rows = [check_initial(q, cp) for q in primes(args.max_q) for cp in (0,1,4)]
    assert set().union(*(set(r['header_values']) for r in rows)) == set(range(5)), 'missing d0 lane'
    check_guard(args.integrated)
    q = model.SECP256K1_Q
    directed = [1, q-1] + [2**k for k in range(256)]
    secp_rows = [check_initial(q, cp, xs=directed) for cp in (0,1,4)]
    controls = {}
    if args.controls:
        for control in ('wrong_header','dropped_rotation'):
            bad = check_initial(127, 4, control)
            assert bad > 0, ('control escaped',control)
            controls[control] = bad
    result = dict(stage='initial',rows=rows, secp_rows=secp_rows, head=subprocess.check_output(
                      ['git','-C',str(Path(__file__).resolve().parent.parent),'rev-parse','HEAD'],text=True).strip(),
                  primes=len(primes(args.max_q)),
                  cases=sum(r['cases'] for r in rows), controls=controls,
                  codec=codec, integrated=args.integrated,
                  status='pass')
    args.out.mkdir(parents=True,exist_ok=True)
    result.update(source_identity())
    (args.out/'initial.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','secp_rows')}),flush=True)


if __name__ == '__main__':
    main()
