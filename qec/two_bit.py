"""The four-symbol tick on a held offset, g = [g0, b].

    g0 = 1:         S = 2S + R mod q, then swap R and S on b
    g0 = 0, b = 0:  S = 2S mod q
    g0 = 0, b = 1:  S = 4S mod q

docs/mux-proof.md states and proves what follows. Credit: the offset
T = S + R between a doubling and the following subtract is the representation of section VI of
arXiv:2609.05625. The tick below keeps that offset as an unreduced integer, so nothing in it is
negative.

After the sweep in point_add.py, every B1 is followed by a subtract or ends the pass. A B1 then
adds R and does no reduction, leaving H = 2T + R, and the subtract after it makes 2H - R = 4T + R.

Add and shift now have complementary controls, so one rolling XOR wire D, the sum of the x lines
so far, lets one CCX carry both: a pair of neighbouring word bits takes D, or on J its own XOR.

H's bit 0 is R's, so it is not stored. Three wires are held between ticks: [this cell is B1],
which the tick makes from its own sw with an X and two CX and the next tick reads as held, in
place of computing it with a CCX, and H's bits n and n + 1, which a B1 moves off the word onto two
clean wires (0 on every other lane) and the next tick takes as its word's top two wires.
"""
from .arith import Mod, _maj, _selected_carry, cswap, gt_into
from . import arith
from functools import lru_cache

# Fixed secp selector circuit. Combo bit0 is 1, bits1..5 are
# k0..k4; the other entry selects one earlier product, or None.
_JUMP4_SELECTOR_TABLES = (
    (((41,None),(35,None)),((49,0),(6,0)),((0,1),(24,0)),
     ((49,2),(10,0)),((41,0),(37,None)),((37,4),(35,None))),
    (((41,None),(35,None)),((0,0),(37,None)),((49,None),(0,1)),
     ((24,0),(20,1)),((0,3),(49,0)),((37,1),(35,0))),
)
_JUMP4_SECP_Q = 2**256-2**32-977
_JUMP4_MONOS = tuple(m for m in range(1,16) if m.bit_count()>=2)


def _jump4_solve(basis,target):
    rows = []
    for j,v in enumerate(basis):
        combo = 1<<j
        for rv,rc in rows:
            if v^rv<v:
                v,combo = v^rv,combo^rc
        if v:
            rows.append((v,combo))
            rows.sort(reverse=True)
    combo = 0
    for rv,rc in rows:
        if target^rv<target:
            target,combo = target^rv,combo^rc
    if target:
        raise ValueError("Jump-4 selector table does not span addend")
    return combo


def _jump4_rotate(C,word,controls,marks):
    """Left rotation, inverse of the right-strip reflection network."""
    n = len(word)
    base = min(range(min(2,n)),key=lambda b:sum(
        sum(i<(m+b-i)%n for i in range(n)) for m in marks))
    for ctrl,mark in reversed(list(zip(controls,marks))):
        for i in reversed(range(n)):
            j = (mark+base-i)%n
            if i<j:
                cswap(C,ctrl,[word[i]],[word[j]])


def _jump4_swap(C,g,b,R,S):
    (sw,) = C.alloc(1)
    C.ccx(g,b,sw)
    cswap(C,sw,R,S)
    C.ccx(g,b,sw)
    C.free([sw])


class _Jump4Reduction:
    """Gate-built Jump-4 coefficient reduction, using ab-godder's arith.add (PR #9, PR #28).

    Fixed secp selector data replaces monomials only in the middle cell.
    Quiet columns of both masks share one Prefix across both carry chains;
    spans are derived from those masks, without a selector or modulus search.
    """
    def __init__(self,q,scale=False,tables=None):
        self.q,self.n,self.scale = q,q.bit_length(),scale
        self.f = (1<<self.n)-q
        if q%16!=15 or 17*self.f>=1<<self.n:
            raise ValueError("unsupported Jump-4 optimized modulus")
        self.w = ((16 if scale else 17)*self.f).bit_length()
        self.fixed = q==_JUMP4_SECP_Q and not scale
        self.tables = _JUMP4_SELECTOR_TABLES if tables is None else tables
        domain = range(16 if scale else 17)
        def vec(fn):
            return sum(1<<k for k in domain if fn(k))
        free = [vec(lambda k:1)]+[vec(lambda k,i=i:(k>>i)&1) for i in range(5)]
        self.masks = []
        for chain in range(2):
            if self.fixed:
                products = []
                for ops in self.tables[chain]:
                    operands = []
                    for combo,idx in ops:
                        if not isinstance(combo,int) or not 0<=combo<64 or (idx is not None and not 0<=idx<len(products)):
                            raise ValueError("invalid Jump-4 affine selector operand")
                        value = 0
                        for j,t in enumerate(free):
                            if combo>>j&1:
                                value ^= t
                        if idx is not None:
                            value ^= products[idx]
                        operands.append(value)
                    if len(operands)!=2:
                        raise ValueError("invalid Jump-4 selector product")
                    products.append(operands[0]&operands[1])
                if len(products)!=6:
                    raise ValueError("Jump-4 requires six fixed products")
                basis = free+products
            else:
                basis = free+[vec(lambda k,m=m:k&m==m) for m in _JUMP4_MONOS]
            masks = [0]*len(basis)
            for i in range(self.w):
                target = vec(lambda k,i=i:(((k+(chain==0))*self.f)>>i)&1)
                combo = _jump4_solve(basis,target)
                # Independently re-evaluate the chosen truth-table mask.
                rebuilt = 0
                for j,t in enumerate(basis):
                    if combo>>j&1:
                        rebuilt ^= t
                        masks[j] |= 1<<i
                if rebuilt!=target:
                    raise ValueError("Jump-4 selector reconstruction mismatch")
            self.masks.append(tuple(masks))
        # A quiet bit's selected addend is zero for every selector state in
        # BOTH chains. Never infer eligibility from one convenient quotient.
        used=0
        for masks in self.masks:
            for mask in masks:used |= mask
        # The part-1 proof deliberately requires K=0..16 in BOTH chains,
        # even for the four-bit header. Its physical domain is narrower,
        # but do not widen spans using that narrower promise.
        for chain in range(2):
            for k in range(17):used |= (k+(chain==0))*self.f
        self.quiet_runs=[]
        i=1
        while i<self.w:
            j=i
            while j<self.w and not (used>>j)&1:j+=1
            if j-i>=2:self.quiet_runs.append((i,j))
            i=max(i+1,j)

    def and_gate(self,C,ops,base,prods,k):
        ends,made = [],[]
        for combo,idx in ops[k]:
            terms = [base[j] for j in range(1,6) if combo>>j&1]
            if idx is not None:
                terms.append(prods[idx])
            if len(terms)==1 and not combo&1:
                ends.append(terms[0]);continue
            (a,) = C.alloc(1)
            for t in terms:
                C.cx(t,a)
            if combo&1:
                C.x(a)
            ends.append(a);made.append((a,terms,combo&1))
        C.ccx(ends[0],ends[1],prods[k])
        for a,terms,neg in reversed(made):
            if neg:
                C.x(a)
            for t in terms:
                C.cx(t,a)
            C.free([a])

    def make_selectors(self,C,K,chain,one):
        masks = self.masks[chain]
        if self.fixed:
            base = [one]+K
            prods = C.alloc(6)
            ops = self.tables[chain]
            for k in range(6):
                self.and_gate(C,ops,base,prods,k)
            wires = base+prods
            made = ('fixed',ops,base,prods)
        else:
            wires = [one]+K+[None]*len(_JUMP4_MONOS)
            made_gates,monos = [],{}
            for j,m in enumerate(_JUMP4_MONOS,6):
                if not masks[j]:
                    continue
                bits = [i for i in range(4) if m>>i&1]
                prev = K[bits[0]]
                for t in range(1,len(bits)):
                    key = tuple(bits[:t+1])
                    if key not in monos:
                        (x,) = C.alloc(1)
                        C.ccx(prev,K[bits[t]],x)
                        made_gates.append((prev,K[bits[t]],x));monos[key]=x
                    prev = monos[key]
                wires[j] = prev
            made = ('monomial',made_gates)
        used = [j for j,m in enumerate(masks) if m]
        if any(wires[j] is None for j in used):
            raise ValueError("missing Jump-4 selector wire")
        return [wires[j] for j in used],[masks[j] for j in used],made

    def unmake(self,C,made):
        if made[0]=='fixed':
            _,ops,base,prods = made
            for k in reversed(range(6)):
                self.and_gate(C,ops,base,prods,k)
            C.free(prods)
        else:
            for a,b,x in reversed(made[1]):
                C.ccx(a,b,x);C.free([x])

    def chain_up(self,C,L,sels,masks,prefixes):
        chain={0:None}
        def rise(i):
            (c,)=C.alloc(1)
            _selected_carry(C,L[i],sels,masks,i,chain[i],c)
            chain[i+1]=c
        i=0
        for (a,b),P in zip(self.quiet_runs,prefixes):
            for j in range(i,a):rise(j)
            (out,)=C.alloc(1);C.ccx(chain[a],P.top,out)
            chain[b]=out;i=b
        for j in range(i,self.w):rise(j)
        return chain

    def chain_down(self,C,L,sels,masks,chain,write,prefixes):
        def fall(i):
            _selected_carry(C,L[i],sels,masks,i,chain[i],chain[i+1])
            C.free([chain[i+1]])
            if write:
                for s,m in zip(sels,masks):
                    if m>>i&1:C.cx(s,L[i])
                if chain[i] is not None:C.cx(chain[i],L[i])
        i=self.w
        for (a,b),P in reversed(list(zip(self.quiet_runs,prefixes))):
            for j in reversed(range(b,i)):fall(j)
            C.ccx(chain[a],P.top,chain[b]);C.free([chain[b]])
            # First chain is read-only: its Prefix survives the quotient
            # increment. Only the second chain writes/restores/frees it.
            if write:P.write(chain[a])
            i=a
        for j in reversed(range(i)):fall(j)

    def increment(self,C,e,K):
        c = [e]+C.alloc(len(K)-1)
        for i in range(len(K)-1):
            C.ccx(c[i],K[i],c[i+1])
        for i in range(len(K)-1,0,-1):
            C.cx(c[i],K[i]);C.ccx(c[i-1],K[i-1],c[i])
        C.cx(e,K[0]);C.free(c[1:])

    def reduce(self,C,low):
        n,w = self.n,self.w
        L,H,K = low[:w],low[w:n],low[n:]
        (one,) = C.alloc(1)
        C.x(one)
        P = Prefix(C,H)
        quiet = [Prefix(C,L[a:b]) for a,b in self.quiet_runs]
        key = K+[None] if self.scale else K
        sels,masks,made = self.make_selectors(C,key,0,one)
        chain = self.chain_up(C,L,sels,masks,quiet)
        (e,) = C.alloc(1)
        if H:
            C.ccx(P.top,chain[w],e)
        else:
            C.cx(chain[w],e)
        self.chain_down(C,L,sels,masks,chain,False,quiet)
        self.unmake(C,made)
        self.increment(C,e,K)
        sels,masks,made = self.make_selectors(C,key,1,one)
        chain = self.chain_up(C,L,sels,masks,quiet)
        if H:
            C.ccx(P.top,chain[w],e)
        else:
            C.cx(chain[w],e)
        C.free([e])
        P.write(chain[w])
        self.chain_down(C,L,sels,masks,chain,True,quiet)
        self.unmake(C,made)
        C.x(one);C.free([one])
        return low[:n],K

    def clear_low(self,C,D,K,e0,e1,t):
        (anybit,) = C.alloc(1)
        for x in (e0,e1,t):
            C.cx(x,anybit)
        C.cx(D[0],K[0])
        C.ccx(anybit,D[1],K[1]);C.ccx(e1,D[2],K[2]);C.ccx(t,D[3],K[3])
        for x in (e0,e1,t):
            C.cx(x,anybit)
        C.free([anybit])

    def move_top(self,C,h,K,T):
        for i in range(4):
            cswap(C,h[i],[K[i+1]],[T])

    def clear_k(self,C,g,e0,e1,R,S,K):
        D = C.alloc(4)
        for i in range(4):
            C.cx(S[i],D[i])
        arith.sub(C,R[:4],D,ctrl=g)
        (t,) = C.alloc(1)
        C.ccx(e0,e1,t)
        self.clear_low(C,D,K,e0,e1,t)
        h = C.alloc(4)
        descriptions = ((h[3],(t,),False),(h[2],(e1,t),False),
                        (h[1],(e0,t),False),(h[0],(e0,e1,t),True))
        for x,srcs,neg in descriptions:
            for s in srcs:
                C.cx(s,x)
            if neg:
                C.x(x)
        y,made,prev = C.alloc(4),[],g
        for i in range(4):
            C.x(D[i]);C.ccx(prev,D[i],y[i]);C.x(D[i])
            made.append((prev,D[i],y[i]));prev=y[i]
        (z,T) = C.alloc(2)
        for i in range(4):
            C.ccx(h[i],y[i],z)
        self.move_top(C,h,K,T)
        gt_into(C,R,S,T,ctrl=z)
        C.free([T])
        for i in range(4):
            C.ccx(h[i],y[i],z)
        C.free([z])
        for prev,d,yi in reversed(made):
            C.x(d);C.ccx(prev,d,yi);C.x(d)
        C.free(y)
        for x,srcs,neg in descriptions:
            if neg:
                C.x(x)
            for s in srcs:
                C.cx(s,x)
        C.free(h);C.ccx(e0,e1,t);C.free([t])
        arith.add(C,R[:4],D,ctrl=g)
        for i in range(4):
            C.cx(S[i],D[i])
        C.free(D)

    def clear_scale(self,C,H,S,K):
        h0,h1,h2 = H
        (t,) = C.alloc(1)
        C.ccx(h0,h1,t)
        f = C.alloc(4)
        descriptions = ((f[0],(h0,h1,h2,t)),(f[1],(h1,h2)),(f[2],(t,h2)),(f[3],(h2,)))
        for x,srcs in descriptions:
            for s in srcs:
                C.cx(s,x)
        for i in range(4):
            C.ccx(f[i],S[i],K[i])
        for x,srcs in descriptions:
            for s in srcs:
                C.cx(s,x)
        C.free(f);C.ccx(h0,h1,t);C.free([t])


@lru_cache(maxsize=None)
def _jump4_reduction(q,scale=False):
    return _Jump4Reduction(q,scale)





class Prefix:
    """AND(bits), and bits += carry, at 3 CCX per bit on about log2(len(bits)) clean wires.

    tops[j] holds a_j = AND(bits[:starts[j]]). When a_j is 1 every wire below starts[j] and
    every lower top reads 1, whether it holds its own value or a lower block's chain, so
    block j's chain can sit on those wires under an X and block lengths double. A chain
    value is only meaningful when a_j is 1, and reaches anything else only through a_j.
    Nothing may read bits between this and write, and write runs top down.
    """

    def __init__(self, C, bits):
        self.C, self.bits = C, bits
        self.blocks = []
        self.released = None
        h = len(bits)
        self.top = bits[0] if h == 1 else None
        if h < 2:
            return
        (self.first,) = C.alloc(1)
        C.ccx(bits[0], bits[1], self.first)
        tops, start = [self.first], 2
        while start < h:
            carriers = bits[:start] + tops[:-1]
            m = min(len(carriers) + 1, h - start)
            x, u = bits[start:start + m], carriers[:m - 1]
            for w in u:
                C.x(w)
            v = x[0]
            for i in range(1, m):
                C.ccx(v, x[i], u[i - 1])
                v = u[i - 1]
            (p,) = C.alloc(1)
            C.ccx(tops[-1], v, p)
            self.blocks.append((tops[-1], p, x, u, v))
            tops.append(p)
            start += m
        self.top = tops[-1]

    def release(self):
        """Clear the top below the last when no wire of the last block's chain holds it or the two
        wires that rebuild it. Only write reads it, and restore rebuilds it first."""
        if len(self.blocks) < 2:
            return
        a, p, _, _, v = self.blocks[-2]
        used = set(self.blocks[-1][3])
        if not {a, p, v} & used:
            self.C.ccx(a, v, p)
            self.C.free([p])
            self.released = len(self.blocks) - 2

    def restore(self):
        if self.released is None:
            return
        a, p, x, u, v = self.blocks[self.released]
        (q,) = self.C.alloc(1)
        self.C.ccx(a, v, q)
        self.blocks[self.released] = a, q, x, u, v
        last = self.blocks[-1]
        self.blocks[-1] = (q,) + last[1:]
        self.released = None

    def write(self, t):
        """bits += t, every wire restored or cleared."""
        C, bits = self.C, self.bits
        assert self.released is None
        if not bits:
            return
        if len(bits) == 1:
            C.cx(t, bits[0])
            return
        for a, p, x, u, v in reversed(self.blocks):
            # p = a*v becomes a*t, the carry into this block.
            C.cx(t, v)
            C.ccx(a, v, p)
            C.cx(t, v)
            for i in reversed(range(1, len(x))):
                below = x[0] if i == 1 else u[i - 2]
                C.ccx(below, x[i], u[i - 1])
                C.ccx(p, below, x[i])
            C.cx(p, x[0])
            for w in reversed(u):
                C.x(w)
            C.ccx(t, a, p)
            C.free([p])
        C.ccx(bits[0], bits[1], self.first)
        C.ccx(t, bits[0], bits[1])
        C.cx(t, bits[0])
        C.free([self.first])


class TwoBitMod(Mod):
    def jump4_cell(self,C,cell,R,S):
        """Canonical (R,S) -> swap_gb(R,2^D S+gR mod q), tape unchanged."""
        if len(cell)!=4 or len(R)!=self.n or len(S)!=self.n:
            raise ValueError("invalid Jump-4 coefficient shape")
        e0,e1,g,b = cell
        if self.q%16!=15 or 17*self.f>=1<<self.n:
            self.moddbl(C,S)
            self._cdbl(C,e0,S)
            for _ in range(2):
                self._cdbl(C,e1,S)
            self.modadd(C,R,S,ctrl=g)
            _jump4_swap(C,g,b,R,S)
            return
        reduction = _jump4_reduction(self.q)
        word = S+C.alloc(4)
        (ex,) = C.alloc(1)
        C.cx(e0,ex);C.cx(e1,ex)
        _jump4_rotate(C,word[:self.n+3],[e0,ex,e1],(3,2,0))
        C.cx(e1,ex);C.cx(e0,ex);C.free([ex])
        pad = C.alloc(5)
        arith.add(C,R[1:]+pad,word,ctrl=g)
        C.free(pad)
        (l0,) = C.alloc(1)
        C.ccx(g,R[0],l0)
        Sp,K = reduction.reduce(C,[l0]+word)
        reduction.clear_k(C,g,e0,e1,R,Sp,K)
        C.free(K)
        _jump4_swap(C,g,b,R,Sp)
        S[:] = Sp


    def jump4_scale(self,C,H,S):
        """Canonical S -> 2^H S mod q, H=0..4, header unchanged."""
        if len(H)!=3 or len(S)!=self.n:
            raise ValueError("invalid Jump-4 scale shape")
        if self.q%16!=15 or 17*self.f>=1<<self.n:
            for bit,repeats in zip(H,(1,2,4)):
                for _ in range(repeats):
                    self._cdbl(C,bit,S)
            return
        reduction = _jump4_reduction(self.q,True)
        word = S+C.alloc(4)
        a,b = C.alloc(2)
        C.cx(H[0],a);C.cx(H[2],a);C.cx(H[0],b);C.cx(H[1],b)
        _jump4_rotate(C,word,[a,b,H[1],H[2]],(4,3,1,0))
        C.cx(H[1],b);C.cx(H[0],b);C.cx(H[2],a);C.cx(H[0],a)
        C.free([a,b])
        Sp,K = reduction.reduce(C,word)
        reduction.clear_scale(C,H,Sp,K)
        C.free(K)
        S[:] = Sp



    def __init__(self, q):
        super().__init__(q)
        n = self.n
        self.f = f = (1 << n) - q
        self.corner = f ^ 2 * f ^ 3 * f
        self.window = (4 * f).bit_length()
        self.merged = 4 * f < 1 << n and self.corner <= 4 * f
        self.runs = self._runs() if self.merged else []

    def _runs(self):
        """Runs of two or more window bits above bit 0 on which every reduction adds the same bit:
        0 on a quiet run, 1 for every k >= 1 on a ones run. As (start, end, ones), low to high."""
        f, w = self.f, self.window
        used = f | 2 * f | 3 * f | 4 * f | self.corner
        ones = f & 2 * f & 3 * f & 4 * f
        kind = [None] + ['ones' if ones >> i & 1 else None if used >> i & 1 else 'quiet'
                         for i in range(1, w)]
        runs, start = [], 1
        for i in range(2, w + 1):
            if i == w or kind[i] != kind[start]:
                if kind[start] and i - start >= 2:
                    runs.append((start, i, kind[start] == 'ones'))
                start = i
        return runs

    def _run_prefixes(self, C, L):
        """One Prefix per run. A ones run takes 2^h - 1 + c = 2^h - (1 - c), a decrement by 1 - c,
        which is an increment of the complement, so its bits sit under an X from here to the write."""
        prefixes = []
        for a, b, ones in self.runs:
            if ones:
                for x in L[a:b]:
                    C.x(x)
            prefixes.append(Prefix(C, L[a:b]))
        return prefixes

    def _bridge(self, C, selectors, masks, a, carry, top, out):
        """out ^= s ^ (carry ^ s)*top, the carry out of the run at a, whose addend bits all equal s.

        s is the XOR of the selectors set at a. On a quiet run s = 0. On a ones run the complement
        is incremented by 1 - carry, and the run carries out unless that increment wraps. When s = 0
        on a ones run the whole addend is 0, so carry = 0 and top is not read."""
        active = [sel for sel, mask in zip(selectors, masks) if mask >> a & 1]
        for sel in active:
            C.cx(sel, carry)
        C.ccx(carry, top, out)
        for sel in active:
            C.cx(sel, carry)
            C.cx(sel, out)

    def step(self, C, g, R, S):
        if self.merged:
            self.tick(C, None, g, None, R, S)
            return
        g0, b = g
        sw = self._begin(C, g0, b)
        self._cdbl(C, b, S)
        C.cx(sw, b)
        self.dbl_add(C, g0, R, S)
        cswap(C, sw, R, S)
        self._end(C, g0, b, sw)

    def _begin(self, C, g0, b):
        """b holds J and a fresh wire holds sw."""
        (sw,) = C.alloc(1)
        C.ccx(g0, b, sw)
        C.cx(sw, b)
        return sw

    def _end(self, C, g0, b, sw):
        """b holds b again; release sw."""
        C.ccx(g0, b, sw)
        C.free([sw])

    def _cdbl(self, C, bit, s):
        """s = (1 + bit)*s mod q. moddbl fixes zero, so park s when bit is 0."""
        parked = C.alloc(self.n)
        C.x(bit)
        cswap(C, bit, s, parked)
        C.x(bit)
        self.moddbl(C, s)
        C.x(bit)
        cswap(C, bit, s, parked)
        C.x(bit)
        C.free(parked)

    def _perches(self, selectors, masks):
        """Bits whose carry out can sit on their selector, as {bit: selector}."""
        perch, read = {}, set()
        for i in reversed(range(1, self.window)):
            on = [s for s, mask in zip(selectors, masks) if mask >> i & 1]
            free = [s for s in on if s not in read]
            if free and not any(a <= i < b for a, b, _ in self.runs):
                perch[i] = free[0], [s for s in on if s != free[0]]
            read.update(on)
        return perch

    def _chain_up(self, C, L, selectors, masks, prefixes, perch, take):
        chain = {0: None}

        def rise(i):
            if i in perch:
                (s, rest), c = perch[i], chain[i]
                for other in rest:
                    C.cx(other, s)
                C.cx(s, L[i])
                C.cx(s, c)
                C.ccx(c, L[i], s)
                chain[i + 1] = s
            else:
                chain[i + 1] = take()
                _selected_carry(C, L[i], selectors, masks, i, chain[i], chain[i + 1])

        i = 0
        for (a, b, _), M in zip(self.runs, prefixes):
            for j in range(i, a):
                rise(j)
            chain[b] = take()
            self._bridge(C, selectors, masks, a, chain[a], M.top, chain[b])
            i = b
        for j in range(i, self.window):
            rise(j)
        return chain

    def _chain_down(self, C, L, selectors, masks, prefixes, perch, chain, drop, write,
                    keep_first=False):
        """keep_first leaves carry 1 on its wire for the caller to clear."""

        def fall(i):
            if i in perch:
                (s, rest), c = perch[i], chain[i]
                C.ccx(c, L[i], s)
                C.cx(s, c)
                C.cx(s, L[i])
                for other in reversed(rest):
                    C.cx(other, s)
            elif i == 0 and keep_first:
                pass
            else:
                _selected_carry(C, L[i], selectors, masks, i, chain[i], chain[i + 1])
                drop(chain[i + 1])
            if write and i:
                for selector, mask in zip(selectors, masks):
                    if mask >> i & 1:
                        C.cx(selector, L[i])
                C.cx(chain[i], L[i])

        i = self.window
        for (a, b, ones), M in reversed(list(zip(self.runs, prefixes))):
            for j in reversed(range(b, i)):
                fall(j)
            self._bridge(C, selectors, masks, a, chain[a], M.top, chain[b])
            drop(chain[b])
            if write:
                active = [sel for sel, mask in zip(selectors, masks) if mask >> a & 1]
                for sel in active:
                    C.cx(sel, chain[a])
                M.restore()
                M.write(chain[a])
                for sel in active:
                    C.cx(sel, chain[a])
                if ones:
                    for x in L[a:b]:
                        C.x(x)
            i = a
        for j in reversed(range(i)):
            fall(j)

    def _mark_b1(self, C, prev, p):
        for w in prev:
            C.x(w)
        C.ccx(prev[0], prev[1], p)
        for w in prev:
            C.x(w)

    def _drop_after_b1(self, C, prev, p):
        if prev is not None:
            self._mark_b1(C, prev, p)
            C.free([p])

    def _add_or_shift(self, C, J, a, word, held_lane=True, g0=None):
        """On J, word = 2*word. Off J, word += a >> 1, through word's top two bits.

        word[n] must be 0 on J: a J cell never follows a B1 (D1), so a J lane is never held.
        held_lane=False promises that no lane is held, so word[n] = 0 on every lane.
        With g0, a B1 lane's sum bits n and n + 1 leave the word for two clean wires, returned as
        (hn, hb) and 0 on every other lane, and the word's top bits n - 1 and n end 0 there."""
        n = self.n
        (cin, D, t) = C.alloc(3)
        ops, tgt = a[1:], word[:n - 1]
        carries = [cin] + ops[:-1]
        for i in range(n - 1):
            _maj(C, carries[i], tgt[i], ops[i])
            C.cx(carries[i], D)
        c, top, mid, hi = ops[-1], word[n - 1], word[n], word[n + 1]
        # t = c * top is the carry out of bit n - 1 off J. On J it is junk, but there mid = 0, so
        # neither hi's update nor mid's needs a J gate on t.
        C.ccx(c, top, t)
        if held_lane:
            C.ccx(t, mid, hi)
        # mid ^= t off J; on J, mid = 0 takes top, J's top move.
        C.cx(t, top)
        C.ccx(J, top, mid)
        C.cx(t, top)
        C.cx(t, mid)
        if g0 is None:
            C.ccx(c, top, t)
        else:
            # hn = B1*(top ^ c), with B1 = 1 ^ g0 ^ J on the g0 wire, and top ^= hn leaves c there.
            # t's clear then reads top ^ B1: c*top off B1, and c*(c ^ 1) = 0 on B1, where t keeps
            # the carry c*top, H's bit n + 1.
            (hn,) = C.alloc(1)
            C.x(g0)
            C.cx(J, g0)
            C.cx(c, top)
            C.ccx(g0, top, hn)
            C.cx(c, top)
            C.cx(hn, top)
            C.cx(g0, top)
            C.ccx(c, top, t)
            C.cx(g0, top)
            C.cx(J, g0)
            C.x(g0)
        # top ^= c ^ D off J: its carry, and the cancel of the D that pair n - 2 hands it. On J top
        # becomes 0, read through mid, which holds top's old value there.
        C.cx(c, D)
        C.cx(D, top)
        C.cx(mid, D)
        C.ccx(J, D, top)
        C.cx(mid, D)
        C.cx(c, D)
        if g0 is not None:
            C.cx(t, mid)
        for i in reversed(range(n - 1)):
            c, u, v, o = carries[i], word[i], word[i + 1], ops[i]
            C.ccx(c, u, o)
            C.cx(o, u)
            # u and v both take D, or on J their own XOR. Then c still holds x_i, for D.
            C.cx(v, u)
            C.cx(D, v)
            C.cx(u, D)
            C.ccx(J, D, v)
            C.cx(u, D)
            C.cx(v, u)
            C.cx(c, D)
            C.cx(o, c)
        if g0 is None:
            C.free([cin, D, t])
            return None
        C.free([cin, D])
        return hn, t

    def _last_tick(self, C, prev, g, a, s, held):
        """Cell 0's tick, where g0 implies b. The pass ends with R = 0, so on entry a J or B1 lane
        has R = 0 and a subtract lane has 2T + R = 0 mod q, or 2H - R = 0 mod q when held. The
        result on a subtract lane is then R itself, made from S alone with no add:

            plain:  S = -2T mod q = ~(V + (k + 1)f - 1) for V = 2T, k = floor(V / q)
            held:   S = 2H mod q
            J:      S = 4T mod q        B1:  S = 2T mod q

        after which R ^= g0*S clears R in place of the swap. V + kf = r + e*2^n for r = V mod q,
        so the plain lane's sum overflows exactly when e is set, as the other lanes' does, and
        its complement is q - r. That is R, except on the lane T = R = 0, where it is q."""
        n, f, w = self.n, self.f, self.window
        g0, b = g
        p, spill, high = held
        if prev is None:
            C.free([p])
            p = None
        word = s + [spill, high]
        C.cx(g0, b)
        for i in reversed(range(n)):
            cswap(C, b, [word[i]], [word[i + 1]])
        if p is not None:
            # A held word is H - R0, with bit 0 clear.
            C.ccx(p, a[0], word[0])
        (l0,) = C.alloc(1)
        low = [l0] + word[:n - 1]
        L, H = low[:w], low[w:]
        k0, k1, k2 = word[n - 1:n + 2]
        (one,) = C.alloc(1)
        C.x(one)
        P, M = Prefix(C, H), self._run_prefixes(C, L)
        spare = H[:] if len(H) > 1 else []

        def alloc():
            (c,) = C.alloc(1)
            return c

        def take():
            if not spare:
                return alloc()
            c = spare.pop()
            C.x(c)
            return c

        def drop(c):
            if c in H:
                C.x(c)
                spare.append(c)
            else:
                C.free([c])

        def plain():
            """Toggle the g0 wire between g0 and [plain subtract lane] = g0 ^ p."""
            if p is not None:
                C.cx(p, g0)

        (three,) = C.alloc(1)
        C.ccx(k0, k1, three)
        selectors = [one, k0, k1, three]
        masks = [f, f ^ 2 * f, f ^ 3 * f, self.corner ^ 4 * f]
        chain = self._chain_up(C, L, selectors, masks, M, {}, take)
        (e,) = C.alloc(1)
        if H:
            C.ccx(P.top, chain[w], e)
        else:
            C.cx(chain[w], e)
        self._chain_down(C, L, selectors, masks, M, {}, chain, drop, False)
        C.x(one)
        C.ccx(three, e, k2)
        C.ccx(e, k1, three)
        C.free([one])
        one = three
        C.ccx(k0, e, k1)
        C.cx(e, k0)
        # A plain lane has k <= 1 and adds (k + 1)f - 1, which is f ^ 1 or 2f ^ 3.
        plain()
        (ck0,) = C.alloc(1)
        C.ccx(g0, k0, ck0)
        selectors = [one, k0, k1, k2, g0, ck0]
        masks = [self.corner, f, 2 * f, 4 * f, f ^ 1, 2 * f ^ 2]
        perch = self._perches(selectors, masks)
        chain = self._chain_up(C, L, selectors, masks, M, perch, alloc)
        if H:
            C.ccx(P.top, chain[w], e)
        else:
            C.cx(chain[w], e)
        C.free([e])
        P.write(chain[w])
        self._chain_down(C, L, selectors, masks, M, perch, chain, drop, True)
        C.ccx(g0, k0, ck0)
        C.free([ck0])
        for x in word[:n]:
            C.cx(g0, x)
        plain()

        if p is not None:
            cswap(C, p, [k1], [k2])
        # one = k0*k1 becomes z = (p ^ wire k2)*~k0, which is [k is 0 or 4] on a held lane and 0
        # elsewhere, as in tick with p for g0.
        z = one
        if p is not None:
            C.cx(p, k1)
        C.ccx(k0, k1, z)
        if p is not None:
            C.cx(p, k1)
            C.cx(p, z)
        C.cx(k2, z)

        for r, x in zip(a, [k0] + word[:n - 1]):
            C.ccx(g0, x, r)
        # R is now 0, or q on the lane T = R = 0, where S is q too. R's top bit flags that lane.
        q = self.q
        for i, (r, x) in enumerate(zip(a, [k0] + word[:n - 1])):
            C.cx(r, x)
            if i < n - 1 and q >> i & 1:
                C.cx(a[n - 1], r)
        plain()
        C.ccx(g0, a[n - 1], k1)
        C.ccx(g0, k1, a[n - 1])
        # Wire k1 holds the flag on a plain lane and k's bit 2 on a held lane. With R = 0,
        # [S >> 1 > R >> 1] is [S >= 2]: it clears bit 2 under z, and the flag is
        # [S is 0] = ~S0 * ~[S >= 2] on a plain lane.
        (zc,) = C.alloc(1)
        C.x(k0)
        C.ccx(g0, k0, zc)
        C.cx(zc, k1)
        C.cx(zc, z)
        gt_into(C, low[1:], a[1:], k1, ctrl=z)
        C.cx(zc, z)
        C.ccx(g0, k0, zc)
        C.x(k0)
        C.free([zc])
        plain()
        if p is not None:
            C.cx(k2, p)
            C.x(k0)
            C.ccx(p, k0, z)
            C.x(k0)
            C.cx(k2, p)
            # 2H = S + kq and H's bit 0 is S's, so k = f*(S - 2*S0) mod 4.
            if not f >> 1 & 1:
                C.cx(k0, low[1])
            C.ccx(p, low[1], k2)
            if not f >> 1 & 1:
                C.cx(k0, low[1])
        C.free([z])
        self._drop_after_b1(C, prev, p)
        C.ccx(b, low[1], k1)
        if f >> 1 & 1:
            C.ccx(b, k0, k1)
        C.cx(g0, b)
        C.free([l0, spill, high])
        s[:] = s[-1:] + s[:-1]
        return None

    def tick(self, C, prev, g, nxt, a, s, pebble=False, held=None, cell0=False, tau=None):
        """One tick of a swept tape. prev is the cell before, or None on the first tick.
        nxt is the next cell's g0 wire, or None on the last tick, whose B1 reduces normally.

        held is what the tick before handed over: [prev is B1], then H's bits n and n + 1 on a held
        lane, which become word[n] and word[n + 1], and 0 elsewhere. All three are 0 when prev is
        None.
        cell0 says this is cell 0, where g0 implies b (Euclid's tick 0 runs on U = q > x, so its
        subtract always swaps). It has its own tick, _last_tick.
        tau, when given, is a wire with S = tau*R on entry and no held lane: the tick after the
        pass's terminal cell, with tau that cell's g0.
        Returns the three wires to hand to the next tick, or None when nxt is None.
        sw and [prev is B1] each stay on one wire for the whole tick."""
        if cell0:
            return self._last_tick(C, prev, g, a, s, held)
        n, f, w = self.n, self.f, self.window
        g0, b = g
        p, spill, high = held
        if prev is None:
            C.free([p])
            p = None
        word = s + [spill, high]
        if tau is not None:
            (rtop,) = C.alloc(1)
            C.cx(s[n - 1], rtop)
        sw = self._begin(C, g0, b)
        if p is not None:
            for x in word:
                C.cx(p, x)
        hand = self._add_or_shift(C, b, a, word, held_lane=prev is not None,
                                  g0=None if nxt is None else g0)
        if p is not None:
            for x in word:
                C.cx(p, x)
        (l0,) = C.alloc(1)
        C.ccx(g0, a[0], l0)
        low = [l0] + word[:n - 1]
        L, H = low[:w], low[w:]
        k0, k1, k2 = word[n - 1:n + 2]

        # one = 0 only on a B1, which is the tick's inert lane. b holds J here, so this needs no CCX.
        (one,) = C.alloc(1)
        if nxt is None:
            C.x(one)
        else:
            C.cx(g0, one)
            C.cx(b, one)
            # The add-or-shift moved H's bits n and n + 1 off a B1 lane's word for the next tick.
            hn, hb = hand
        P, M = Prefix(C, H), self._run_prefixes(C, L)

        # With at most one H bit, P reads a data wire or nothing, and there is nothing to borrow.
        spare = H[:] if len(H) > 1 else []

        def alloc():
            (c,) = C.alloc(1)
            return c

        def take():
            if not spare:
                return alloc()
            c = spare.pop()
            C.x(c)
            return c

        def drop(c):
            if c in H:
                C.x(c)
                spare.append(c)
            else:
                C.free([c])

        (three,) = C.alloc(1)
        C.ccx(k0, k1, three)
        selectors = [one, k0, k1, three]
        masks = [f, f ^ 2 * f, f ^ 3 * f, self.corner ^ 4 * f]
        chain = self._chain_up(C, L, selectors, masks, M, {}, take)
        (e,) = C.alloc(1)
        if H:
            C.ccx(P.top, chain[w], e)
        else:
            C.cx(chain[w], e)
        self._chain_down(C, L, selectors, masks, M, {}, chain, drop, False)
        if nxt is None:
            C.x(one)
        else:
            C.cx(b, one)
            C.cx(g0, one)
        C.cx(sw, b)
        C.ccx(three, e, k2)
        # three = K0*K1 becomes k0*k1 = K0*K1 ^ e*K1 for k = K + e, so it reads k1 before k1 changes.
        C.ccx(e, k1, three)
        C.free([one])
        one = three
        C.ccx(k0, e, k1)
        C.cx(e, k0)
        selectors, masks = [one, k0, k1, k2], [self.corner, f, 2 * f, 4 * f]
        perch = self._perches(selectors, masks)
        # Every low carry stays on its own wire until its fall, and e stays on its own wire.
        chain = self._chain_up(C, L, selectors, masks, M, perch, alloc)
        if H:
            C.ccx(P.top, chain[w], e)
        else:
            C.cx(chain[w], e)
        C.free([e])
        P.write(chain[w])
        self._chain_down(C, L, selectors, masks, M, perch, chain, drop, True,
                         keep_first=p is not None)
        # Carry 1 of the second chain is l0*k0, and l0 = R0 on a held lane, so it is bit_one's AND.
        k0r0 = chain[1] if p is not None else None

        # On a held lane k1 and k2 change wires, so the compare's target is k's top bit there.
        if p is not None:
            cswap(C, p, [k1], [k2])
        def mark_even(z):
            C.cx(k2, g0)
            C.x(k0)
            C.ccx(g0, k0, z)
            C.x(k0)
            C.cx(k2, g0)

        # one = k0*k1 becomes z = (g0 ^ wire k2)*~k0 with one CCX. Wire k2 now holds k's bit 1 on a
        # held lane and 0 elsewhere, and k0*k2 = 0 because k <= 4, so k0*k1 = k0*(wire k1 ^ wire k2)
        # and z ^ k0*k1 = k0*(wire k1 ^ g0) ^ g0 ^ wire k2.
        z = one
        C.cx(g0, k1)
        C.ccx(k0, k1, z)
        C.cx(g0, k1)
        C.cx(g0, z)
        C.cx(k2, z)
        if tau is None:
            gt_into(C, a[1:], low[1:], k1, ctrl=z)
        else:
            # V = (2*tau + 1)*R here, so an even k is 2 only when V = 3R >= 2q. Then R >= 2q/3,
            # which is at least 2^(n - 1) because 4f < 2^n, and k = 0 has R < q/3 < 2^(n - 1): k's
            # bit 1 is tau*R's top bit, copied from S before the add.
            C.ccx(z, rtop, k1)
            C.ccx(tau, a[n - 1], rtop)
            C.free([rtop])
        mark_even(z)
        C.free([z])
        if p is not None:
            # S = R + k*f mod 4 names k's bit 1, which now sits on k2.
            def bit_one(m):
                C.cx(low[1], m)
                C.cx(a[1], m)
                if f >> 1 & 1:
                    C.cx(k0, m)

            bit_one(k0r0)
            C.ccx(p, k0r0, k2)
            bit_one(k0r0)
            C.ccx(L[0], k0, k0r0)
            C.free([k0r0])
        self._drop_after_b1(C, prev, p)
        C.cx(sw, b)
        C.ccx(b, low[1], k1)
        if f >> 1 & 1:
            C.ccx(b, k0, k1)
        C.cx(l0, k0)
        C.cx(sw, b)
        C.ccx(g0, a[0], l0)
        C.free([l0, spill, high])
        s[:] = s[-1:] + s[:-1]
        cswap(C, sw, a, s)
        if nxt is None:
            self._end(C, g0, b, sw)
            return None
        # [this cell is B1] = (1 ^ g0)(1 ^ b) = 1 ^ g0 ^ b ^ sw, the next tick's held.
        C.x(sw)
        C.cx(g0, sw)
        C.cx(b, sw)
        return [sw, hn, hb]
